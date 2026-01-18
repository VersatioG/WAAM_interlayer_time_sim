"""
Asynchronous State Manager for WAAM Thermal Simulation

This module extends the simulation state manager with asynchronous logging capabilities
using threading and RAM buffering to minimize performance impact during simulation.

Features:
- Asynchronous HDF5 writes using background thread
- RAM buffer to decouple simulation from I/O operations
- Thread-safe queue for log entries
- Automatic buffer flushing
- Graceful shutdown with flush-on-close
"""

import os
import h5py
import numpy as np
import json
from pathlib import Path
import threading
import queue
from collections import deque
import time

# Import base functionality
from simulation_state_manager import (
    extract_parameters_from_globals,
    compare_parameters,
    SimulationStateManager
)


class AsyncSimulationStateManager(SimulationStateManager):
    """
    Asynchronous simulation state manager with threaded I/O.
    
    Uses a background thread to write data to HDF5 while the simulation continues.
    Data is buffered in RAM and written in batches for optimal performance.
    """
    
    def __init__(self, filename, current_params, total_nodes, buffer_size=100):
        """
        Initialize async state manager.
        
        Args:
            filename: HDF5 file path
            current_params: Simulation parameters
            total_nodes: Total number of thermal nodes
            buffer_size: Number of timesteps to buffer before forcing flush
        """
        super().__init__(filename, current_params, total_nodes)
        
        # Threading components
        self.log_queue = queue.Queue(maxsize=200)  # Queue for log entries
        self.writer_thread = None
        self.shutdown_event = threading.Event()
        self.flush_event = threading.Event()
        
        # Buffer management
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.log_counter = 0
        
        # Statistics
        self.writes_completed = 0
        self.writes_pending = 0
        
    def initialize_or_load(self):
        """
        Setup the HDF5 file and start the background writer thread.
        """
        start_layer, resume_state = super().initialize_or_load()
        
        # Start background writer thread
        self._start_writer_thread()
        
        return start_layer, resume_state
    
    def _start_writer_thread(self):
        """Start the background writer thread."""
        if self.writer_thread is None or not self.writer_thread.is_alive():
            self.shutdown_event.clear()
            self.writer_thread = threading.Thread(
                target=self._writer_loop,
                name="HDF5-Writer",
                daemon=False  # Non-daemon to ensure proper shutdown
            )
            self.writer_thread.start()
    
    def _writer_loop(self):
        """
        Background thread loop that writes buffered data to HDF5.
        Runs until shutdown_event is set.
        """
        local_buffer = []
        
        while not self.shutdown_event.is_set() or not self.log_queue.empty():
            try:
                # Try to get an item with timeout
                try:
                    item = self.log_queue.get(timeout=0.1)
                except queue.Empty:
                    # No item available, check if we should flush
                    if local_buffer and (self.flush_event.is_set() or len(local_buffer) >= self.buffer_size):
                        self._write_buffer(local_buffer)
                        local_buffer.clear()
                        self.flush_event.clear()
                    continue
                
                # Handle different item types
                if item is None:
                    # Shutdown signal
                    break
                elif item[0] == 'log_step':
                    # Add to local buffer
                    local_buffer.append(item)
                    self.writes_pending = len(local_buffer)
                    
                    # Flush if buffer is full
                    if len(local_buffer) >= self.buffer_size:
                        self._write_buffer(local_buffer)
                        local_buffer.clear()
                        
                elif item[0] == 'mark_layer_complete':
                    # Flush buffer first, then mark layer
                    if local_buffer:
                        self._write_buffer(local_buffer)
                        local_buffer.clear()
                    self._mark_layer_complete_immediate(item[1], item[2])
                    
                elif item[0] == 'flush':
                    # Immediate flush request
                    if local_buffer:
                        self._write_buffer(local_buffer)
                        local_buffer.clear()
                
                self.log_queue.task_done()
                
            except Exception as e:
                print(f"Error in writer thread: {e}")
                import traceback
                traceback.print_exc()
        
        # Final flush on shutdown
        if local_buffer:
            self._write_buffer(local_buffer)
    
    def _write_buffer(self, buffer):
        """
        Write a batch of log entries to HDF5.
        
        Args:
            buffer: List of log entries to write
        """
        if not buffer or self.file is None:
            return
        
        try:
            # Get current size
            current_idx = self.file[self.DS_TIME].shape[0]
            batch_size = len(buffer)
            new_size = current_idx + batch_size
            
            # Resize all datasets once
            self.file[self.DS_TIME].resize((new_size,))
            self.file[self.DS_LAYER_IDX].resize((new_size,))
            self.file[self.DS_TEMPS].resize((new_size, self.total_nodes))
            self.file[self.DS_ACTIVE].resize((new_size, self.total_nodes))
            self.file[self.DS_LEVEL_TYPE].resize((new_size, self.total_nodes))
            self.file[self.DS_RAD_AREAS].resize((new_size, self.total_nodes))
            
            if self.DS_SUMMARY_TIME in self.file:
                self.file[self.DS_SUMMARY_TIME].resize((new_size,))
                self.file[self.DS_SUMMARY_BP].resize((new_size,))
                self.file[self.DS_SUMMARY_TABLE].resize((new_size,))
                self.file[self.DS_SUMMARY_LAYERS].resize((new_size, self.file[self.DS_SUMMARY_LAYERS].shape[1]))
            
            # Write all entries in batch
            for i, item in enumerate(buffer):
                idx = current_idx + i
                _, time_val, layer_idx, node_matrix_data, summary_data = item
                
                # Write main data
                self.file[self.DS_TIME][idx] = time_val
                self.file[self.DS_LAYER_IDX][idx] = layer_idx
                self.file[self.DS_TEMPS][idx, :] = node_matrix_data['temperatures']
                self.file[self.DS_ACTIVE][idx, :] = node_matrix_data['active_mask']
                self.file[self.DS_LEVEL_TYPE][idx, :] = node_matrix_data['level_type']
                self.file[self.DS_RAD_AREAS][idx, :] = node_matrix_data['radiation_areas']
                
                # Write summary data
                if summary_data and self.DS_SUMMARY_TIME in self.file:
                    self.file[self.DS_SUMMARY_TIME][idx] = time_val
                    self.file[self.DS_SUMMARY_BP][idx] = summary_data['bp']
                    self.file[self.DS_SUMMARY_TABLE][idx] = summary_data['table']
                    
                    num_recorded = len(summary_data['layers'])
                    total_slots = self.file[self.DS_SUMMARY_LAYERS].shape[1]
                    padded_layers = np.full(total_slots, -1.0, dtype='f4')
                    padded_layers[:num_recorded] = summary_data['layers']
                    self.file[self.DS_SUMMARY_LAYERS][idx, :] = padded_layers
            
            # Update node mapping (use the most recent valid map in the buffer)
            if buffer:
                last_valid_static = None
                for item in reversed(buffer):
                    if item[3].get('layer_idx') is not None:
                        last_valid_static = item[3]
                        break
                
                if last_valid_static:
                    self.file[self.DS_NODE_MAP_LAYER][:] = last_valid_static['layer_idx']
                    self.file[self.DS_NODE_MAP_BEAD][:] = last_valid_static['bead_idx']
                    self.file[self.DS_NODE_MAP_ELEM][:] = last_valid_static['element_idx']
            
            # Flush to disk
            self.file.flush()
            
            self.writes_completed += batch_size
            self.writes_pending = 0
            
        except Exception as e:
            print(f"Error writing buffer to HDF5: {e}")
            import traceback
            traceback.print_exc()
    
    def log_step(self, time, layer_idx, node_matrix, summary_data=None):
        """
        Queue a timestep for asynchronous writing.
        
        Args:
            time: Simulation time
            layer_idx: Current layer index
            node_matrix: NodeMatrix instance
            summary_data: Optional summary dictionary
        """
        if self.file is None:
            return
        
        self.log_counter += 1
        # Only copy static mapping arrays periodically (every buffer_size steps)
        # to save main thread CPU time.
        include_static = (self.log_counter % self.buffer_size == 0) or (self.log_counter == 1)

        # Extract data from node_matrix (copy to avoid threading issues)
        # Note: level_type is active int8, so .copy() is sufficient (no astype needed if already int8)
        node_matrix_data = {
            'temperatures': node_matrix.temperatures[:self.total_nodes].copy(),
            'active_mask': node_matrix.active_mask[:self.total_nodes].astype('i1'),
            'level_type': node_matrix.level_type[:self.total_nodes].copy(), 
            'radiation_areas': node_matrix.radiation_areas[:self.total_nodes].copy(),
        }
        
        if include_static:
            node_matrix_data['layer_idx'] = node_matrix.layer_idx[:self.total_nodes].copy()
            node_matrix_data['bead_idx'] = node_matrix.bead_idx[:self.total_nodes].copy()
            node_matrix_data['element_idx'] = node_matrix.element_idx[:self.total_nodes].copy()
        else:
            node_matrix_data['layer_idx'] = None
            node_matrix_data['bead_idx'] = None
            node_matrix_data['element_idx'] = None
        
        # Queue the log entry
        try:
            self.log_queue.put(
                ('log_step', time, layer_idx, node_matrix_data, summary_data),
                block=False
            )
        except queue.Full:
            # If queue is full, force a flush and retry
            self.flush()
            self.log_queue.put(
                ('log_step', time, layer_idx, node_matrix_data, summary_data),
                block=True,
                timeout=5.0
            )
    
    def mark_layer_complete(self, layer_idx, wait_time=None):
        """
        Queue layer completion marker (ensures all pending writes are done first).
        
        Args:
            layer_idx: Layer index that was completed
            wait_time: Optional wait time for this layer
        """
        try:
            self.log_queue.put(
                ('mark_layer_complete', layer_idx, wait_time),
                block=True,
                timeout=5.0
            )
        except queue.Full:
            print(f"Warning: Queue full when marking layer {layer_idx} complete")
    
    def _mark_layer_complete_immediate(self, layer_idx, wait_time=None):
        """
        Immediately mark layer as complete (called from writer thread).
        """
        if self.file:
            self.file.attrs[self.ATTR_LAST_COMPLETED_LAYER] = layer_idx
            
            if wait_time is not None:
                try:
                    current_waits = json.loads(self.file.attrs.get(self.ATTR_WAIT_TIMES, '[]'))
                    current_waits.append(wait_time)
                    self.file.attrs[self.ATTR_WAIT_TIMES] = json.dumps(current_waits)
                except Exception as e:
                    print(f"Error saving wait time: {e}")
            
            self.file.flush()
    
    def flush(self):
        """
        Force an immediate flush of all buffered data.
        Blocks until all pending writes are complete.
        """
        if self.writer_thread and self.writer_thread.is_alive():
            # Signal flush and wait
            self.flush_event.set()
            
            # Give the thread time to process
            for _ in range(50):  # Wait up to 5 seconds
                if self.log_queue.qsize() == 0 and self.writes_pending == 0:
                    break
                time.sleep(0.1)
    
    def close(self):
        """
        Close the state manager and ensure all data is written.
        Blocks until background thread completes.
        """
        if self.writer_thread and self.writer_thread.is_alive():
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for queue to empty
            self.log_queue.join()
            
            # Wait for thread to finish
            self.writer_thread.join(timeout=10.0)
            
            if self.writer_thread.is_alive():
                print("Warning: Writer thread did not shut down cleanly")
        
        # Close the file
        if self.file:
            self.file.close()
            self.file = None


def create_state_manager(filename, params, total_nodes, async_mode=True):
    """
    Factory function to create appropriate state manager.
    
    Args:
        filename: HDF5 file path
        params: Simulation parameters
        total_nodes: Total number of nodes
        async_mode: If True, use async manager (default)
    
    Returns:
        SimulationStateManager or AsyncSimulationStateManager instance
    """
    if async_mode:
        return AsyncSimulationStateManager(filename, params, total_nodes)
    else:
        return SimulationStateManager(filename, params, total_nodes)
