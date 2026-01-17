"""
Simulation State Manager for WAAM Thermal Simulation

This module provides functionality to save and resume thermal simulations,
enabling restart capability if the simulation is interrupted.

Features:
- Save simulation state periodically to file
- Resume simulation from saved state
- Validate parameter consistency (with exception for NUMBER_OF_LAYERS)
"""

import numpy as np
import pickle
import json
from pathlib import Path


class SimulationState:
    """Container for simulation state that can be saved/loaded."""
    
    def __init__(self):
        # Simulation parameters (for validation)
        self.parameters = {}
        
        # Current simulation state
        self.current_time = 0.0
        self.current_layer = 0
        self.logging_counter = 0
        
        # Node matrix state
        self.temperatures = None
        self.active_mask = None
        self.layer_idx = None
        self.bead_idx = None
        self.element_idx = None
        self.level_type = None
        self.masses = None
        self.areas = None
        self.radiation_areas = None
        
        # Active WAAM indices
        self.active_waam_indices = None
        self.bp_covered_area = 0.0
        
        # Table state
        self.table_indices = []
        self.table_idx = None
        self.table_bp_contact_idx = None
        self.table_grid = None
        self.table_node_dims = None
        self.table_subdivisions = None
        
        # Logged data
        self.time_log = []
        self.temp_layers_log = []
        self.temp_bp_log = []
        self.temp_table_log = []
        self.wait_times = []


def extract_parameters_from_globals():
    """
    Extract all relevant simulation parameters from the global scope.
    These will be saved with the state for validation on resume.
    """
    import Thermal_Sim as ts
    
    params = {
        # Simulation settings
        'DT': ts.DT,
        'LOGGING_FREQUENCY': ts.LOGGING_FREQUENCY,
        'LOGGING_EVERY_N_STEPS': ts.LOGGING_EVERY_N_STEPS,
        
        # Discretization
        'N_LAYERS_AS_BEADS': ts.N_LAYERS_AS_BEADS,
        'N_LAYERS_WITH_ELEMENTS': ts.N_LAYERS_WITH_ELEMENTS,
        'N_ELEMENTS_PER_BEAD': ts.N_ELEMENTS_PER_BEAD,
        
        # WAAM process parameters
        'NUMBER_OF_LAYERS': ts.NUMBER_OF_LAYERS,
        'LAYER_HEIGHT': ts.LAYER_HEIGHT,
        'TRACK_WIDTH': ts.TRACK_WIDTH,
        'TRACK_OVERLAP': ts.TRACK_OVERLAP,
        'NUMBER_OF_TRACKS': ts.NUMBER_OF_TRACKS,
        'TRACK_LENGTH': ts.TRACK_LENGTH,
        'PROCESS_SPEED': ts.PROCESS_SPEED,
        'MELTING_TEMP': ts.MELTING_TEMP,
        'INTERLAYER_TEMP': ts.INTERLAYER_TEMP,
        'ARC_POWER': ts.ARC_POWER,
        'ARC_POWER_CURRENT_FRACTION': ts.ARC_POWER_CURRENT_FRACTION,
        'WIRE_FEED_RATE': ts.WIRE_FEED_RATE,
        'WIRE_DIAMETER': ts.WIRE_DIAMETER,
        
        # Materials
        'MATERIAL_WAAM_NAME': ts.MATERIAL_WAAM_NAME,
        'RHO_WAAM': ts.RHO_WAAM,
        'LAMBDA_WAAM': ts.LAMBDA_WAAM,
        'EPSILON_WAAM': ts.EPSILON_WAAM,
        'EPSILON_WAAM_LIQUID': ts.EPSILON_WAAM_LIQUID,
        
        # Base plate
        'BP_LENGTH': ts.BP_LENGTH,
        'BP_WIDTH': ts.BP_WIDTH,
        'BP_THICKNESS': ts.BP_THICKNESS,
        'MATERIAL_BP_NAME': ts.MATERIAL_BP_NAME,
        'RHO_BP': ts.RHO_BP,
        'LAMBDA_BP': ts.LAMBDA_BP,
        'EPSILON_BP': ts.EPSILON_BP,
        
        # Table
        'TABLE_LENGTH': ts.TABLE_LENGTH,
        'TABLE_WIDTH': ts.TABLE_WIDTH,
        'TABLE_THICKNESS': ts.TABLE_THICKNESS,
        'MATERIAL_TABLE_NAME': ts.MATERIAL_TABLE_NAME,
        'RHO_TABLE': ts.RHO_TABLE,
        'LAMBDA_TABLE': ts.LAMBDA_TABLE,
        'EPSILON_TABLE': ts.EPSILON_TABLE,
        'TABLE_DISCRETIZATION_MODE': ts.TABLE_DISCRETIZATION_MODE,
        'N_TABLE_X': ts.N_TABLE_X,
        'N_TABLE_Y': ts.N_TABLE_Y,
        'N_TABLE_Z': ts.N_TABLE_Z,
        
        # Environment
        'AMBIENT_TEMP': ts.AMBIENT_TEMP,
        'CONTACT_BP_TABLE': ts.CONTACT_BP_TABLE,
        'ALPHA_CONTACT': ts.ALPHA_CONTACT,
        'STEFAN_BOLTZMANN': ts.STEFAN_BOLTZMANN,
    }
    
    return params


def save_simulation_state(state, filename):
    """
    Save simulation state to file using pickle serialization.
    
    SECURITY NOTE: This function uses pickle for serialization. Only load state files
    from trusted sources, as pickle can execute arbitrary code when loading. Never load
    state files from untrusted or unknown sources.
    
    Args:
        state: SimulationState object
        filename: Path to save file
    """
    filepath = Path(filename)
    
    # Create backup if file exists
    if filepath.exists():
        backup_path = filepath.with_suffix('.backup')
        if backup_path.exists():
            backup_path.unlink()
        filepath.rename(backup_path)
    
    # Save state using pickle for binary efficiency
    with open(filepath, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Simulation state saved to {filename}")


def load_simulation_state(filename):
    """
    Load simulation state from file.
    
    Args:
        filename: Path to state file
        
    Returns:
        SimulationState object or None if file doesn't exist
    """
    filepath = Path(filename)
    
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print(f"Simulation state loaded from {filename}")
        return state
    except Exception as e:
        print(f"Error loading state from {filename}: {e}")
        return None


def compare_parameters(saved_params, current_params, ignore_keys=None):
    """
    Compare saved and current parameters to check if they match.
    
    Args:
        saved_params: Dictionary of saved parameters
        current_params: Dictionary of current parameters
        ignore_keys: List of parameter keys to ignore in comparison
        
    Returns:
        tuple: (match, differences)
            match: Boolean indicating if parameters match
            differences: Dictionary of differing parameters
                - Key: parameter name
                - Value: tuple of (saved_value, current_value)
                - Special case: ('missing', saved_value, None) if key not in current_params
    """
    if ignore_keys is None:
        ignore_keys = []
    
    differences = {}
    
    for key in saved_params:
        if key in ignore_keys:
            continue
            
        if key not in current_params:
            # Special tuple format for missing parameters
            differences[key] = ('missing', saved_params[key], None)
            continue
        
        saved_val = saved_params[key]
        current_val = current_params[key]
        
        # Compare with tolerance for floating point values
        if isinstance(saved_val, (int, float)) and isinstance(current_val, (int, float)):
            if not np.isclose(saved_val, current_val, rtol=1e-9):
                differences[key] = (saved_val, current_val)
        else:
            if saved_val != current_val:
                differences[key] = (saved_val, current_val)
    
    match = len(differences) == 0
    return match, differences


def check_state_file_compatibility(filename):
    """
    Check if saved state file is compatible with current parameters.
    
    Args:
        filename: Path to state file
        
    Returns:
        tuple: (compatible, state, differences)
            compatible: Boolean indicating compatibility
            state: Loaded SimulationState object or None
            differences: Dictionary of parameter differences
    """
    state = load_simulation_state(filename)
    
    if state is None:
        return False, None, {}
    
    current_params = extract_parameters_from_globals()
    
    # NUMBER_OF_LAYERS can be changed (to extend or shorten simulation)
    ignore_keys = ['NUMBER_OF_LAYERS']
    
    match, differences = compare_parameters(state.parameters, current_params, ignore_keys)
    
    if not match:
        print("\nParameter mismatch detected:")
        for key, (saved_val, current_val) in differences.items():
            print(f"  {key}: saved={saved_val}, current={current_val}")
    
    return match, state, differences


def create_state_from_node_matrix(node_matrix, current_time, current_layer, 
                                   logging_counter, time_log, temp_layers_log, 
                                   temp_bp_log, temp_table_log, wait_times):
    """
    Create a SimulationState object from current simulation data.
    
    Args:
        node_matrix: NodeMatrix object
        current_time: Current simulation time [s]
        current_layer: Current layer index
        logging_counter: Current logging counter
        time_log: List of logged times [s]
        temp_layers_log: List of logged layer temperatures [°C]
        temp_bp_log: List of logged base plate temperatures [°C]
        temp_table_log: List of logged table temperatures [°C]
        wait_times: List of wait times per layer [s]
        
    Returns:
        SimulationState object
    """
    state = SimulationState()
    
    # Save parameters for validation
    state.parameters = extract_parameters_from_globals()
    
    # Save current simulation progress
    state.current_time = current_time
    state.current_layer = current_layer
    state.logging_counter = logging_counter
    
    # Save node matrix state (deep copy arrays)
    state.temperatures = node_matrix.temperatures.copy()
    state.active_mask = node_matrix.active_mask.copy()
    state.layer_idx = node_matrix.layer_idx.copy()
    state.bead_idx = node_matrix.bead_idx.copy()
    state.element_idx = node_matrix.element_idx.copy()
    state.level_type = node_matrix.level_type.copy()
    state.masses = node_matrix.masses.copy()
    state.areas = node_matrix.areas.copy()
    state.radiation_areas = node_matrix.radiation_areas.copy()
    
    # Save active WAAM indices
    state.active_waam_indices = node_matrix.active_waam_indices.copy()
    state.bp_covered_area = node_matrix.bp_covered_area
    
    # Save table state
    state.table_indices = node_matrix.table_indices.copy()
    state.table_idx = node_matrix.table_idx
    state.table_bp_contact_idx = node_matrix.table_bp_contact_idx
    if node_matrix.table_grid is not None:
        state.table_grid = node_matrix.table_grid.copy()
    state.table_node_dims = node_matrix.table_node_dims
    state.table_subdivisions = node_matrix.table_subdivisions
    
    # Save logged data
    state.time_log = list(time_log)
    state.temp_layers_log = [list(layer_temps) for layer_temps in temp_layers_log]
    state.temp_bp_log = list(temp_bp_log)
    state.temp_table_log = list(temp_table_log)
    state.wait_times = list(wait_times)
    
    return state


def restore_node_matrix_from_state(state, node_matrix):
    """
    Restore NodeMatrix object from saved state.
    Handles cases where the node_matrix size may differ (e.g., when extending simulation).
    
    When extending a simulation (increasing NUMBER_OF_LAYERS), the new node_matrix
    will be larger than the saved state. This function copies the saved data into
    the first portion of the arrays. New nodes beyond copy_size remain at their
    initialized values (typically AMBIENT_TEMP for temperatures, False for active_mask).
    
    Args:
        state: SimulationState object
        node_matrix: NodeMatrix object to restore into
    """
    # Determine the size to copy (minimum of saved and current)
    saved_size = len(state.temperatures)
    current_size = len(node_matrix.temperatures)
    copy_size = min(saved_size, current_size)
    
    # Restore arrays (only up to the saved size)
    # New nodes beyond copy_size keep their initialization values
    node_matrix.temperatures[:copy_size] = state.temperatures[:copy_size]
    node_matrix.active_mask[:copy_size] = state.active_mask[:copy_size]
    node_matrix.layer_idx[:copy_size] = state.layer_idx[:copy_size]
    node_matrix.bead_idx[:copy_size] = state.bead_idx[:copy_size]
    node_matrix.element_idx[:copy_size] = state.element_idx[:copy_size]
    node_matrix.level_type[:copy_size] = state.level_type[:copy_size]
    node_matrix.masses[:copy_size] = state.masses[:copy_size]
    node_matrix.areas[:copy_size] = state.areas[:copy_size]
    node_matrix.radiation_areas[:copy_size] = state.radiation_areas[:copy_size]
    
    # Restore active indices
    node_matrix.active_waam_indices = state.active_waam_indices.copy()
    node_matrix.bp_covered_area = state.bp_covered_area
    
    # Restore table state
    node_matrix.table_indices = state.table_indices.copy()
    node_matrix.table_idx = state.table_idx
    node_matrix.table_bp_contact_idx = state.table_bp_contact_idx
    if state.table_grid is not None:
        node_matrix.table_grid = state.table_grid.copy()
    node_matrix.table_node_dims = state.table_node_dims
    node_matrix.table_subdivisions = state.table_subdivisions
