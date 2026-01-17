"""
Simulation State Manager for WAAM Thermal Simulation

This module provides functionality to save and resume thermal simulations using HDF5.
It handles continuous logging of simulation data (temperatures, active nodes) and 
manages the resume capability by rolling back to the last successfully completed layer.

Features:
- Continuous logging to HDF5 (append mode)
- Efficient storage of temperature matrices and active masks
- Resume capability:
  - Detects last COMPLETED layer
  - Removes partial data from interrupted layers
  - Restores full simulation state (NodeMatrix) from file
- Parameter validation to ensure consistency
"""

import os
import h5py
import numpy as np
import json
from pathlib import Path

# =============================================================================
# PARAMETER EXTRACTION & UTILS
# =============================================================================

def extract_parameters_from_globals(ts_globals):
    """
    Extract relevant simulation parameters from the Thermal_Sim globals.
    Use this to capture the configuration for validation.
    """
    # Define the keys we care about
    keys = [
        # Simulation settings
        'DT', 'LOGGING_FREQUENCY', 'LOGGING_EVERY_N_STEPS',
        
        # Discretization
        'N_LAYERS_AS_BEADS', 'N_LAYERS_WITH_ELEMENTS', 'N_ELEMENTS_PER_BEAD',
        
        # WAAM process parameters
        'NUMBER_OF_LAYERS', 'LAYER_HEIGHT', 'TRACK_WIDTH', 'TRACK_OVERLAP', 
        'NUMBER_OF_TRACKS', 'TRACK_LENGTH', 'PROCESS_SPEED', 'MELTING_TEMP', 
        'INTERLAYER_TEMP', 'ARC_POWER', 'ARC_POWER_CURRENT_FRACTION', 
        'WIRE_FEED_RATE', 'WIRE_DIAMETER',
        
        # Materials
        'MATERIAL_WAAM_NAME', 'RHO_WAAM', 'LAMBDA_WAAM', 'EPSILON_WAAM', 'EPSILON_WAAM_LIQUID',
        
        # Base plate
        'BP_LENGTH', 'BP_WIDTH', 'BP_THICKNESS', 'MATERIAL_BP_NAME', 
        'RHO_BP', 'LAMBDA_BP', 'EPSILON_BP',
        
        # Table
        'TABLE_LENGTH', 'TABLE_WIDTH', 'TABLE_THICKNESS', 'MATERIAL_TABLE_NAME', 
        'RHO_TABLE', 'LAMBDA_TABLE', 'EPSILON_TABLE', 
        'TABLE_DISCRETIZATION_MODE', 'N_TABLE_X', 'N_TABLE_Y', 'N_TABLE_Z',
        
        # Environment
        'AMBIENT_TEMP', 'CONTACT_BP_TABLE', 'ALPHA_CONTACT', 'STEFAN_BOLTZMANN',
    ]
    
    params = {}
    for key in keys:
        if hasattr(ts_globals, key):
            params[key] = getattr(ts_globals, key)
        elif key in ts_globals: # Handle dict if passed instead of module
             params[key] = ts_globals[key]
            
    return params


def compare_parameters(saved_params, current_params, ignore_keys=None):
    """
    Compare saved and current parameters.
    Returns (match_boolean, diff_dict)
    """
    if ignore_keys is None:
        ignore_keys = []
    
    differences = {}
    for key, saved_val in saved_params.items():
        if key in ignore_keys:
            continue
            
        if key not in current_params:
            differences[key] = (saved_val, 'MISSING')
            continue
            
        curr_val = current_params[key]
        
        # Numeric comparison with tolerance
        if isinstance(saved_val, (int, float)) and isinstance(curr_val, (int, float)):
            if not np.isclose(saved_val, curr_val, rtol=1e-5):
                differences[key] = (saved_val, curr_val)
        elif saved_val != curr_val:
            differences[key] = (saved_val, curr_val)
            
    return len(differences) == 0, differences


# =============================================================================
# HDF5 STATE MANAGER
# =============================================================================

class SimulationStateManager:
    def __init__(self, filename, current_params, total_nodes):
        self.filename = Path(filename)
        self.current_params = current_params
        self.total_nodes = total_nodes
        self.file = None

        # Dataset Names
        self.DS_TIME = 'time'
        self.DS_LAYER_IDX = 'layer_indices'
        self.DS_TEMPS = 'temperatures'
        self.DS_ACTIVE = 'active_mask'
        self.DS_LEVEL_TYPE = 'level_type'
        self.DS_RAD_AREAS = 'radiation_areas'
        
        # Node Mapping Datasets (Static/Topology)
        self.DS_NODE_MAP_LAYER = 'node_map_layer'
        self.DS_NODE_MAP_BEAD = 'node_map_bead'
        self.DS_NODE_MAP_ELEM = 'node_map_element'

        self.DS_MASSES = 'masses' # Static usually, but good to check or save once
        self.DS_AREAS = 'areas'   # Static
        
        # Summary Datasets (for plotting)
        self.DS_SUMMARY_TIME = 'summary_time'
        self.DS_SUMMARY_LAYERS = 'summary_layers'
        self.DS_SUMMARY_BP = 'summary_bp'
        self.DS_SUMMARY_TABLE = 'summary_table'
        
        # Attribute Names
        self.ATTR_PARAMS = 'parameters'
        self.ATTR_LAST_COMPLETED_LAYER = 'last_completed_layer'
        self.ATTR_WAIT_TIMES = 'wait_times'

    def initialize_or_load(self):
        """
        Setup the HDF5 file.
        If exists: Checks params, prunes unfinished layers, returns resume state.
        If new: Creates datasets.
        
        Returns:
            start_layer_idx (int): The layer index to start/resume from.
            resume_state (dict or None): Dictionary to restore NodeMatrix if resuming.
        """
        if self.filename.exists():
            return self._handle_existing_file()
        else:
            self._create_new_file()
            return 0, None

    def _create_new_file(self):
        """Create new HDF5 file with unlimited time dimension."""
        print(f"Creating new state file: {self.filename}")
        self.file = h5py.File(self.filename, 'w')
        
        # Save parameters
        # Convert dict to JSON string for attribute storage (simpler than types)
        # Handle numpy types in params if any
        serializable_params = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                               for k,v in self.current_params.items()}
        self.file.attrs[self.ATTR_PARAMS] = json.dumps(serializable_params)
        self.file.attrs[self.ATTR_LAST_COMPLETED_LAYER] = -1 # None completed
        self.file.attrs[self.ATTR_WAIT_TIMES] = json.dumps([])
        
        # Create Datasets
        # Shape: (Time, Nodes)
        # MaxShape: (Unlimited, Unlimited) to allow extending time AND nodes (if extending sim)
        chunk_size_time = 100
        
        self.file.create_dataset(self.DS_TIME, shape=(0,), maxshape=(None,), 
                                 dtype='f8', chunks=(chunk_size_time,))
        
        self.file.create_dataset(self.DS_LAYER_IDX, shape=(0,), maxshape=(None,), 
                                 dtype='i4', chunks=(chunk_size_time,))
        
        self.file.create_dataset(self.DS_TEMPS, shape=(0, self.total_nodes), maxshape=(None, None),
                                 dtype='f4', chunks=(chunk_size_time, self.total_nodes), compression="gzip")
        
        self.file.create_dataset(self.DS_ACTIVE, shape=(0, self.total_nodes), maxshape=(None, None),
                                 dtype='i1', chunks=(chunk_size_time, self.total_nodes), compression="gzip") # Store as int8
        
        self.file.create_dataset(self.DS_LEVEL_TYPE, shape=(0, self.total_nodes), maxshape=(None, None),
                                 dtype='i1', chunks=(chunk_size_time, self.total_nodes), compression="gzip")
        
        self.file.create_dataset(self.DS_RAD_AREAS, shape=(0, self.total_nodes), maxshape=(None, None),
                                 dtype='f4', chunks=(chunk_size_time, self.total_nodes), compression="gzip")
        
        # Static Node Mapping (Topological indices)
        self.file.create_dataset(self.DS_NODE_MAP_LAYER, shape=(self.total_nodes,), dtype='i4')
        self.file.create_dataset(self.DS_NODE_MAP_BEAD, shape=(self.total_nodes,), dtype='i4')
        self.file.create_dataset(self.DS_NODE_MAP_ELEM, shape=(self.total_nodes,), dtype='i4')
        
        # Summary Datasets (Plotting)
        num_layers = self.current_params.get('NUMBER_OF_LAYERS', 1)
        self.file.create_dataset(self.DS_SUMMARY_TIME, shape=(0,), maxshape=(None,), dtype='f8', chunks=(chunk_size_time,))
        self.file.create_dataset(self.DS_SUMMARY_BP, shape=(0,), maxshape=(None,), dtype='f4', chunks=(chunk_size_time,))
        self.file.create_dataset(self.DS_SUMMARY_TABLE, shape=(0,), maxshape=(None,), dtype='f4', chunks=(chunk_size_time,))
        self.file.create_dataset(self.DS_SUMMARY_LAYERS, shape=(0, num_layers), maxshape=(None, num_layers), 
                                 dtype='f4', chunks=(chunk_size_time, num_layers))
        
        self.file.flush()

    def _handle_existing_file(self):
        """Check existing file, prune partials, extract state."""
        print(f"Checking existing state file: {self.filename}")
        self.file = h5py.File(self.filename, 'r+') # Read/Write
        
        # 1. Parameter Check
        try:
            saved_params = json.loads(self.file.attrs[self.ATTR_PARAMS])
        except Exception as e:
            self.file.close()
            raise ValueError(f"Parameters missing or corrupt in '{self.filename}': {e}. "
                             f"Delete the file to start a new simulation.")
            
        # Allow NUMBER_OF_LAYERS to change
        match, diff = compare_parameters(saved_params, self.current_params, ignore_keys=['NUMBER_OF_LAYERS'])
        if not match:
            print("❌ Parameters differ from saved state:")
            for k, v in diff.items():
                print(f"   {k}: Saved={v[0]}, Current={v[1]}")
            self.file.close()
            raise ValueError(f"Simulation parameters do not match the existing file '{self.filename}'. "
                             f"To start a new simulation, delete or rename the file.")
        
        # 2. Pruning Logic
        # Strategy: "Wiedereinstieg nur zum Start eines Layers"
        # We rely on 'last_completed_layer' attribute.
        last_completed = self.file.attrs.get(self.ATTR_LAST_COMPLETED_LAYER, -1)
        
        print(f"✓ Parameters match. Last completed layer in file: {last_completed}")
        
        # If simulation was complete (last_completed == NUM_LAYERS -1) AND we extended layers, we resume.
        # Logic holds: last_completed is safe.
        
        if last_completed == -1:
            print("No completed layers found. Restarting from Layer 0.")
            self._truncate_datasets(0)
            return 0, None
            
        # Find the index in datasets corresponding to the LAST step of last_completed_layer
        layer_indices = self.file[self.DS_LAYER_IDX][:]
        
        # We want to keep steps where layer_index <= last_completed
        valid_indices = np.where(layer_indices <= last_completed)[0]
        
        if len(valid_indices) == 0:
             # Should not happen if last_completed > -1 but safety first
             self._truncate_datasets(0)
             return 0, None
             
        cut_point = valid_indices[-1] + 1
        
        # Truncate if there are extra steps (partial next layer)
        if cut_point < len(layer_indices):
            print(f"Pruning unfinished layer data (Steps {cut_point} to {len(layer_indices)})...")
            self._truncate_datasets(cut_point)
            
        # 3. Extract Resume State
        # Get the VERY LAST valid frame
        # Make sure to handle array dimensions if they differ from self.total_nodes
        saved_nodes = self.file[self.DS_TEMPS].shape[1]
        
        # Read the slice
        temps = self.file[self.DS_TEMPS][cut_point-1]
        active = self.file[self.DS_ACTIVE][cut_point-1]
        level_type = self.file[self.DS_LEVEL_TYPE][cut_point-1]
        rad_areas = self.file[self.DS_RAD_AREAS][cut_point-1]
        time = self.file[self.DS_TIME][cut_point-1]
        
        resume_state = {
            'temperatures': temps,
            'active_mask': active.astype(bool),
            'level_type': level_type.astype('i1'),
            'radiation_areas': rad_areas,
            'time': time,
            'layer_idx': last_completed + 1
        }

        # Load summary logs
        if self.DS_SUMMARY_TIME in self.file:
            # We must only load up to cut_point
            sm_time = self.file[self.DS_SUMMARY_TIME][:cut_point]
            sm_bp = self.file[self.DS_SUMMARY_BP][:cut_point]
            sm_table = self.file[self.DS_SUMMARY_TABLE][:cut_point]
            sm_layers_raw = self.file[self.DS_SUMMARY_LAYERS][:cut_point]
            
            # Reconstruct list of lists for layers
            # sm_layers_raw is [Time x MaxLayers]
            # Since NUMBER_OF_LAYERS is constant, we can just slice based on top layer?
            # Actually Thermal_Sim expects [ [L1], [L1, L2], ... ]
            # We can just filter out -1 or NaN (assuming we used a fill value).
            # Default HDF5 fill is 0. But 0 is valid temp. Let's assume -1 for inactive.
            
            reconstructed_layers = []
            layer_ids_in_file = self.file[self.DS_LAYER_IDX][:cut_point]
            for step_idx in range(cut_point):
                curr_top = layer_ids_in_file[step_idx]
                reconstructed_layers.append(list(sm_layers_raw[step_idx, :curr_top+1]))
            
            resume_state['history'] = {
                'time': list(sm_time),
                'bp': list(sm_bp),
                'table': list(sm_table),
                'layers': reconstructed_layers
            }
        
        # Extract wait times
        try:
            wait_times = json.loads(self.file.attrs.get(self.ATTR_WAIT_TIMES, '[]'))
            # Prune wait times if they exceed the completed layers
            # last_completed is 0-based index. If last_completed=0 (Layer 0 done), we have 1 wait time? 
            # Wait time is calculated AFTER layer is done. So we should have len(wait_times) == last_completed + 1
            if len(wait_times) > last_completed + 1:
                wait_times = wait_times[:last_completed + 1]
            resume_state['wait_times'] = wait_times
        except:
            resume_state['wait_times'] = []

        # 4. Handle Node Count Mismatch (Extension)
        if self.total_nodes > saved_nodes:
             print(f"Extending simulation nodes: {saved_nodes} -> {self.total_nodes}")
             
             # Resize HDF5 datasets columns
             for name in [self.DS_TEMPS, self.DS_ACTIVE, self.DS_RAD_AREAS]:
                 dset = self.file[name]
                 dset.resize((dset.shape[0], self.total_nodes))
                 
             # Pad the resume_state vectors with defaults
             pad_width = self.total_nodes - saved_nodes
             ambient = self.current_params.get('AMBIENT_TEMP', 25.0)
             resume_state['temperatures'] = np.pad(resume_state['temperatures'], (0, pad_width), constant_values=ambient)
             resume_state['active_mask'] = np.pad(resume_state['active_mask'], (0, pad_width), constant_values=False)
             resume_state['radiation_areas'] = np.pad(resume_state['radiation_areas'], (0, pad_width), constant_values=0.0)
             
        elif self.total_nodes < saved_nodes:
             print("Warning: Current node count is SMALLER than saved state. truncating nodes.")
             resume_state['temperatures'] = resume_state['temperatures'][:self.total_nodes]
             resume_state['active_mask'] = resume_state['active_mask'][:self.total_nodes]
             resume_state['radiation_areas'] = resume_state['radiation_areas'][:self.total_nodes]
             
             # Resize HDF5 datasets columns to shrink
             for name in [self.DS_TEMPS, self.DS_ACTIVE, self.DS_RAD_AREAS]:
                 dset = self.file[name]
                 dset.resize((dset.shape[0], self.total_nodes))

        next_layer = int(last_completed + 1)
        print(f"Resuming simulation at start of Layer {next_layer}")
        return next_layer, resume_state

    def _truncate_datasets(self, size):
        """Truncate all time-dependent datasets to 'size'."""
        names = [self.DS_TIME, self.DS_LAYER_IDX, self.DS_TEMPS, self.DS_ACTIVE, self.DS_RAD_AREAS,
                 self.DS_SUMMARY_TIME, self.DS_SUMMARY_BP, self.DS_SUMMARY_TABLE, self.DS_SUMMARY_LAYERS]
        for name in names:
             if name in self.file:
                current_shape = list(self.file[name].shape)
                current_shape[0] = size
                self.file[name].resize(tuple(current_shape))
        self.file.flush()

    def log_step(self, time, layer_idx, node_matrix, summary_data=None):
        """
        Append a single timestep to the file.
        summary_data: dict with 'bp', 'table', 'layers' (list of maxes)
        """
        if self.file is None:
            return

        # Current index
        idx = self.file[self.DS_TIME].shape[0]
        new_size = idx + 1
        
        # Resize dimensions (Time axis)
        self.file[self.DS_TIME].resize((new_size,))
        self.file[self.DS_LAYER_IDX].resize((new_size,))
        self.file[self.DS_TEMPS].resize((new_size, self.total_nodes))
        self.file[self.DS_ACTIVE].resize((new_size, self.total_nodes))
        self.file[self.DS_LEVEL_TYPE].resize((new_size, self.total_nodes))
        self.file[self.DS_RAD_AREAS].resize((new_size, self.total_nodes))
        
        # Write data
        self.file[self.DS_TIME][idx] = time
        self.file[self.DS_LAYER_IDX][idx] = layer_idx
        self.file[self.DS_TEMPS][idx, :] = node_matrix.temperatures[:self.total_nodes]
        self.file[self.DS_ACTIVE][idx, :] = node_matrix.active_mask[:self.total_nodes].astype('i1')
        self.file[self.DS_LEVEL_TYPE][idx, :] = node_matrix.level_type[:self.total_nodes].astype('i1')
        self.file[self.DS_RAD_AREAS][idx, :] = node_matrix.radiation_areas[:self.total_nodes]

        # Update Node Mapping (Static properties, but populated dynamically)
        self.file[self.DS_NODE_MAP_LAYER][:] = node_matrix.layer_idx[:self.total_nodes]
        self.file[self.DS_NODE_MAP_BEAD][:] = node_matrix.bead_idx[:self.total_nodes]
        self.file[self.DS_NODE_MAP_ELEM][:] = node_matrix.element_idx[:self.total_nodes]

        # Log Summary
        if summary_data and self.DS_SUMMARY_TIME in self.file:
            self.file[self.DS_SUMMARY_TIME].resize((new_size,))
            self.file[self.DS_SUMMARY_BP].resize((new_size,))
            self.file[self.DS_SUMMARY_TABLE].resize((new_size,))
            self.file[self.DS_SUMMARY_LAYERS].resize((new_size, self.file[self.DS_SUMMARY_LAYERS].shape[1]))
            
            self.file[self.DS_SUMMARY_TIME][idx] = time
            self.file[self.DS_SUMMARY_BP][idx] = summary_data['bp']
            self.file[self.DS_SUMMARY_TABLE][idx] = summary_data['table']
            
            # Pad layers_maxes to fixed width
            num_recorded = len(summary_data['layers'])
            total_slots = self.file[self.DS_SUMMARY_LAYERS].shape[1]
            padded_layers = np.full(total_slots, -1.0, dtype='f4')
            padded_layers[:num_recorded] = summary_data['layers']
            self.file[self.DS_SUMMARY_LAYERS][idx, :] = padded_layers

    def mark_layer_complete(self, layer_idx, wait_time=None):
        """Update the last completed layer attribute and log wait time."""
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
            

    def close(self):
        if self.file:
            self.file.close()

# Helper for compatibility with legacy code calling
def create_state_manager(filename, params, total_nodes):
    return SimulationStateManager(filename, params, total_nodes)

