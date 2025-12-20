"""
Node Temperature Logging Script for WAAM Thermal Simulation

This script runs the thermal simulation and logs the temperature of every individual
node (table, baseplate, WAAM layers) at each logging timestep to an HDF5 file.

The output file contains:
- Timestep values
- Node metadata (ID, type, position)
- Temperature matrix [timesteps x nodes]

Usage:
    python log_node_temperatures.py [--output OUTPUT_FILE] [--layers N]

Example:
    python log_node_temperatures.py --output node_temps.h5 --layers 5
"""

import numpy as np
import h5py
import argparse
from tqdm import tqdm
from scipy.integrate import quad

# Import simulation components from Thermal_Sim
from Thermal_Sim import (
    # Parameters
    DT, LOGGING_EVERY_N_STEPS, N_LAYERS_AS_BEADS, N_LAYERS_WITH_ELEMENTS, N_ELEMENTS_PER_BEAD,
    NUMBER_OF_LAYERS, LAYER_HEIGHT, TRACK_WIDTH, TRACK_OVERLAP, NUMBER_OF_TRACKS, TRACK_LENGTH,
    PROCESS_SPEED, MELTING_TEMP, INTERLAYER_TEMP, ARC_POWER, ARC_POWER_CURRENT_FRACTION,
    WIRE_FEED_RATE, WIRE_DIAMETER,
    RHO_WAAM, LAMBDA_WAAM,
    BP_LENGTH, BP_WIDTH, BP_THICKNESS, RHO_BP, LAMBDA_BP, EPSILON_BP,
    TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS, RHO_TABLE, LAMBDA_TABLE, EPSILON_TABLE,
    TABLE_DISCRETIZATION_MODE, N_TABLE_X, N_TABLE_Y, N_TABLE_Z,
    AMBIENT_TEMP, CONTACT_BP_TABLE, ALPHA_CONTACT, STEFAN_BOLTZMANN,
    # Classes and functions
    NodeMatrix, ThermalModel,
    get_cp_waam, get_cp_bp, get_cp_table, get_epsilon_waam,
    update_temperatures_matrix, calculate_wire_melting_power,
    TYPE_TABLE, TYPE_SPECIAL, TYPE_LAYER, TYPE_BEAD, TYPE_ELEMENT
)


def compute_node_positions(node_matrix):
    """
    Compute 3D positions for all nodes in the simulation.
    
    Returns:
        positions: numpy array of shape (n_nodes, 3) with [x, y, z] positions
        node_types: list of node type strings
    """
    n_nodes = len(node_matrix.temperatures)
    positions = np.zeros((n_nodes, 3), dtype=np.float64)
    node_types = []
    
    # Effective layer width
    effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
    
    for i in range(n_nodes):
        level = node_matrix.level_type[i]
        layer = node_matrix.layer_idx[i]
        bead = node_matrix.bead_idx[i]
        element = node_matrix.element_idx[i]
        
        if level == TYPE_TABLE:
            # Table nodes - use grid position
            pos = node_matrix.get_table_grid_position(i)
            if pos is not None and node_matrix.table_node_dims is not None:
                ix, iy, iz = pos
                dx, dy, dz = node_matrix.table_node_dims
                # Table starts at origin, nodes are at cell centers
                positions[i] = [
                    ix * dx + dx / 2,
                    iy * dy + dy / 2,
                    -(TABLE_THICKNESS - iz * dz - dz / 2)  # Below z=0
                ]
            elif node_matrix.table_idx is not None and i == node_matrix.table_idx:
                 # Single node table
                 positions[i] = [TABLE_LENGTH/2, TABLE_WIDTH/2, -TABLE_THICKNESS/2]
            node_types.append('table')
            
        elif level == TYPE_SPECIAL and i == node_matrix.bp_idx:
            # Base plate - centered on table corner, above table
            positions[i] = [
                BP_LENGTH / 2,
                BP_WIDTH / 2,
                BP_THICKNESS / 2  # Above table (z=0 is table top)
            ]
            node_types.append('baseplate')
            
        elif level == TYPE_LAYER:
            # Layer-level node - centered on layer
            positions[i] = [
                TRACK_LENGTH / 2,
                effective_layer_width / 2,
                BP_THICKNESS + (layer + 0.5) * LAYER_HEIGHT
            ]
            node_types.append('layer')
            
        elif level == TYPE_BEAD:
            # Bead-level node
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            y_offset = 0
            for b in range(bead):
                y_offset += TRACK_WIDTH if b == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            y_offset += bead_width / 2
            
            positions[i] = [
                TRACK_LENGTH / 2,
                y_offset,
                BP_THICKNESS + (layer + 0.5) * LAYER_HEIGHT
            ]
            node_types.append('bead')
            
        elif level == TYPE_ELEMENT:
            # Element-level node
            element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            y_offset = 0
            for b in range(bead):
                y_offset += TRACK_WIDTH if b == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            y_offset += bead_width / 2
            
            positions[i] = [
                (element + 0.5) * element_length,
                y_offset,
                BP_THICKNESS + (layer + 0.5) * LAYER_HEIGHT
            ]
            node_types.append('element')
            
        else:
            # Unknown type
            positions[i] = [0, 0, 0]
            node_types.append('unknown')
    
    return positions, node_types


def compute_node_dimensions(node_matrix):
    """
    Compute dimensions (dx, dy, dz) for visualization blocks.
    
    Returns:
        dimensions: numpy array of shape (n_nodes, 3) with [dx, dy, dz] dimensions
    """
    n_nodes = len(node_matrix.temperatures)
    dimensions = np.zeros((n_nodes, 3), dtype=np.float64)
    
    effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
    
    for i in range(n_nodes):
        level = node_matrix.level_type[i]
        bead = node_matrix.bead_idx[i]
        
        if level == TYPE_TABLE:
            if node_matrix.table_node_dims is not None:
                dx, dy, dz = node_matrix.table_node_dims
                dimensions[i] = [dx, dy, dz]
            elif node_matrix.table_idx is not None and i == node_matrix.table_idx:
                dimensions[i] = [TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS]
            
        elif level == TYPE_SPECIAL and i == node_matrix.bp_idx:
            dimensions[i] = [BP_LENGTH, BP_WIDTH, BP_THICKNESS]
            
        elif level == TYPE_LAYER:
            dimensions[i] = [TRACK_LENGTH, effective_layer_width, LAYER_HEIGHT]
            
        elif level == TYPE_BEAD:
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            dimensions[i] = [TRACK_LENGTH, bead_width, LAYER_HEIGHT]
            
        elif level == TYPE_ELEMENT:
            element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            dimensions[i] = [element_length, bead_width, LAYER_HEIGHT]
    
    return dimensions


def run_simulation_with_logging(num_layers=None, output_file='node_temperatures.h5'):
    """
    Run the thermal simulation and log all node temperatures to HDF5.
    
    Args:
        num_layers: Number of layers to simulate (default: use NUMBER_OF_LAYERS)
        output_file: Path to output HDF5 file
    """
    if num_layers is None:
        num_layers = NUMBER_OF_LAYERS
    
    print(f"Starting simulation with logging...")
    print(f"Output file: {output_file}")
    print(f"Layers: {num_layers}")
    
    # --- Input validation ---
    if N_LAYERS_AS_BEADS > num_layers:
        raise ValueError(f"N_LAYERS_AS_BEADS ({N_LAYERS_AS_BEADS}) must be <= num_layers ({num_layers})")
    
    if N_LAYERS_WITH_ELEMENTS > N_LAYERS_AS_BEADS:
        raise ValueError(f"N_LAYERS_WITH_ELEMENTS ({N_LAYERS_WITH_ELEMENTS}) must be <= N_LAYERS_AS_BEADS ({N_LAYERS_AS_BEADS})")
    
    # Calculate geometry
    effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
    layer_area = effective_layer_width * TRACK_LENGTH
    side_area_layer = 2 * (TRACK_LENGTH + effective_layer_width) * LAYER_HEIGHT
    
    # Bead parameters
    bead_params = {
        'track_width': TRACK_WIDTH,
        'track_length': TRACK_LENGTH,
        'layer_height': LAYER_HEIGHT,
        'n_tracks': NUMBER_OF_TRACKS,
        'overlap': TRACK_OVERLAP,
        'n_elements': N_ELEMENTS_PER_BEAD
    }
    
    # Initialize thermal model
    model = ThermalModel(layer_area, side_area_layer, bead_params)
    
    # Calculate max WAAM nodes
    max_waam_nodes = num_layers * NUMBER_OF_TRACKS * N_ELEMENTS_PER_BEAD
    
    # Calculate table nodes
    if TABLE_DISCRETIZATION_MODE == 0:
        num_table_nodes = 1
        nx, ny, nz = 1, 1, 1
    else:
        nx = N_TABLE_X + (TABLE_DISCRETIZATION_MODE - 1)
        ny = N_TABLE_Y + (TABLE_DISCRETIZATION_MODE - 1)
        nz = N_TABLE_Z + (TABLE_DISCRETIZATION_MODE - 1)
        num_table_nodes = nx * ny * nz
        
    # Initialize node matrix
    node_matrix = NodeMatrix(max_waam_nodes, num_table_nodes, NUMBER_OF_TRACKS, N_ELEMENTS_PER_BEAD)
    
    # Calculate wire melting power
    wire_melting_power = calculate_wire_melting_power(
        WIRE_FEED_RATE, WIRE_DIAMETER, AMBIENT_TEMP, MELTING_TEMP, RHO_WAAM
    )
    effective_arc_power = ARC_POWER - wire_melting_power
    
    # Initialize table
    if TABLE_DISCRETIZATION_MODE == 0:
        # Single table node
        vol_table = TABLE_LENGTH * TABLE_WIDTH * TABLE_THICKNESS
        m_table = vol_table * RHO_TABLE
        table_idx = node_matrix.add_table_node(
            0, mass=m_table, area=TABLE_LENGTH * TABLE_WIDTH, temperature=AMBIENT_TEMP
        )
        node_matrix.table_idx = table_idx
        node_matrix.table_indices = [table_idx]
        node_matrix.table_bp_contact_idx = table_idx
    else:
        # Multi-node table
        node_matrix.initialize_table_grid(
            nx, ny, nz, TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS,
            RHO_TABLE, AMBIENT_TEMP
        )
        # BP contact is at top corner node (0, 0, nz-1)
        node_matrix.table_bp_contact_idx = node_matrix.table_grid[0, 0, nz - 1]
    
    # Initialize base plate
    vol_bp = BP_LENGTH * BP_WIDTH * BP_THICKNESS
    m_bp = vol_bp * RHO_BP
    bp_idx = node_matrix.add_bp_node(
        mass=m_bp, area=BP_LENGTH * BP_WIDTH, temperature=AMBIENT_TEMP
    )
    node_matrix.bp_idx = bp_idx
    
    # Calculate timing
    track_time = TRACK_LENGTH / PROCESS_SPEED
    layer_time = track_time * NUMBER_OF_TRACKS
    
    print(f"Layer geometry: {effective_layer_width*1000:.1f}mm x {TRACK_LENGTH*1000:.1f}mm x {LAYER_HEIGHT*1000:.1f}mm")
    print(f"Layer duration: {layer_time:.1f}s ({NUMBER_OF_TRACKS} tracks at {PROCESS_SPEED*1000:.1f}mm/s)")
    print(f"Arc Power: Total = {ARC_POWER:.1f} W, Wire Melting = {wire_melting_power:.1f} W, Effective = {effective_arc_power:.1f} W")
    
    # Data logging lists (temporary, will be saved to HDF5)
    t_log = []
    temp_snapshots = []  # List of temperature arrays at each log step
    
    # Track node metadata at each snapshot (nodes can be added/removed)
    node_metadata_snapshots = []  # List of (positions, dimensions, types) tuples
    
    t = 0.0
    logging_counter = 0
    wait_times = []
    
    # Main simulation loop
    for i_layer in tqdm(range(num_layers), desc="Simulating layers"):
        current_num_layers = i_layer + 1
        
        # Determine discretization for this layer
        use_beads = (N_LAYERS_AS_BEADS >= 1)
        use_elements = (N_LAYERS_WITH_ELEMENTS >= 1)
        
        # Add beads/elements for the new layer
        for i_track in range(NUMBER_OF_TRACKS):
            # Bead geometry
            if i_track == 0:
                bead_width = TRACK_WIDTH
            else:
                bead_width = TRACK_WIDTH * (1 - TRACK_OVERLAP)
            
            bead_area = bead_width * TRACK_LENGTH
            bead_volume = bead_area * LAYER_HEIGHT
            bead_mass = bead_volume * RHO_WAAM
            
            if use_elements:
                # Add elements for this bead
                element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
                element_area = bead_width * element_length
                element_volume = element_area * LAYER_HEIGHT
                element_mass = element_volume * RHO_WAAM
                element_time = element_length / PROCESS_SPEED
                
                for i_elem in range(N_ELEMENTS_PER_BEAD):
                    new_idx = node_matrix.activate_waam_node(
                        layer_idx=i_layer, bead_idx=i_track, element_idx=i_elem,
                        level_type_str='element', mass=element_mass, area=element_area,
                        temperature=MELTING_TEMP
                    )
                    
                    # Welding time for this element
                    n_steps = int(element_time / DT)
                    for _ in range(n_steps):
                        update_temperatures_matrix(node_matrix, model, 
                                                   is_welding=True, 
                                                   arc_power=effective_arc_power,
                                                   welding_node_idx=new_idx)
                        t += DT
                        logging_counter += 1
                        
                        if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                            t_log.append(t)
                            temp_snapshots.append(np.array(node_matrix.temperatures, dtype=np.float32))
                            positions, types = compute_node_positions(node_matrix)
                            dims = compute_node_dimensions(node_matrix)
                            node_metadata_snapshots.append((positions.copy(), dims.copy(), types.copy()))
            else:
                # Add bead node
                new_idx = node_matrix.activate_waam_node(
                    layer_idx=i_layer, bead_idx=i_track, element_idx=-1,
                    level_type_str='bead', mass=bead_mass, area=bead_area,
                    temperature=MELTING_TEMP
                )
                
                # Welding time for this bead
                n_steps = int(track_time / DT)
                for _ in range(n_steps):
                    update_temperatures_matrix(node_matrix, model,
                                               is_welding=True,
                                               arc_power=effective_arc_power,
                                               welding_node_idx=new_idx)
                    t += DT
                    logging_counter += 1
                    
                    if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                        t_log.append(t)
                        temp_snapshots.append(np.array(node_matrix.temperatures, dtype=np.float32))
                        positions, types = compute_node_positions(node_matrix)
                        dims = compute_node_dimensions(node_matrix)
                        node_metadata_snapshots.append((positions.copy(), dims.copy(), types.copy()))
        
        # Interlayer cooling (dynamic waiting)
        t_wait_start = t
        top_nodes = node_matrix.get_nodes_in_layer(i_layer)
        
        while True:
            if top_nodes:
                top_temps = [node_matrix.temperatures[n] for n in top_nodes]
                max_top_temp = max(top_temps)
                if max_top_temp <= INTERLAYER_TEMP:
                    break
            else:
                break
            
            update_temperatures_matrix(node_matrix, model, is_welding=False)
            t += DT
            logging_counter += 1
            
            if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                t_log.append(t)
                temp_snapshots.append(np.array(node_matrix.temperatures, dtype=np.float32))
                positions, types = compute_node_positions(node_matrix)
                dims = compute_node_dimensions(node_matrix)
                node_metadata_snapshots.append((positions.copy(), dims.copy(), types.copy()))
        
        wait_time = t - t_wait_start
        wait_times.append(wait_time)
        
        # Note: We skip consolidation in the logging script to keep all nodes 
        # for detailed visualization. The standard simulation consolidates 
        # older layers for efficiency, but for visualization purposes we want
        # to see all nodes at their original resolution.
    
    print(f"\nSimulation complete. Total time: {t:.1f}s")
    print(f"Logged {len(t_log)} timesteps with node temperatures")
    
    # Save to HDF5
    print(f"Saving to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        # Metadata
        f.attrs['num_layers'] = num_layers
        f.attrs['dt'] = DT
        f.attrs['logging_interval'] = LOGGING_EVERY_N_STEPS
        f.attrs['total_time'] = t
        f.attrs['ambient_temp'] = AMBIENT_TEMP
        f.attrs['melting_temp'] = MELTING_TEMP
        f.attrs['interlayer_temp'] = INTERLAYER_TEMP
        
        # Timesteps
        f.create_dataset('timesteps', data=np.array(t_log, dtype=np.float32))
        
        # Wait times per layer
        f.create_dataset('wait_times', data=np.array(wait_times, dtype=np.float32))
        
        # Create variable-length datasets for each timestep
        # Since node count can change, we store each snapshot separately
        snapshots_grp = f.create_group('snapshots')
        
        for i, (temps, (positions, dims, types)) in enumerate(zip(temp_snapshots, node_metadata_snapshots)):
            snap_grp = snapshots_grp.create_group(f'{i:06d}')
            snap_grp.create_dataset('temperatures', data=temps)
            snap_grp.create_dataset('positions', data=positions.astype(np.float32))
            snap_grp.create_dataset('dimensions', data=dims.astype(np.float32))
            
            # Store types as encoded integers for efficiency
            type_map = {'table': 0, 'baseplate': 1, 'layer': 2, 'bead': 3, 'element': 4, 'unknown': 5}
            type_codes = np.array([type_map.get(t, 5) for t in types], dtype=np.int8)
            snap_grp.create_dataset('types', data=type_codes)
        
        f.attrs['num_snapshots'] = len(temp_snapshots)
    
    print(f"Saved {len(temp_snapshots)} snapshots to {output_file}")
    print(f"File size: {np.round(h5py.File(output_file, 'r').id.get_filesize() / 1024 / 1024, 2)} MB")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Run WAAM thermal simulation with detailed node temperature logging'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='node_temperatures.h5',
        help='Output HDF5 file path (default: node_temperatures.h5)'
    )
    parser.add_argument(
        '--layers', '-l',
        type=int,
        default=None,
        help=f'Number of layers to simulate (default: {NUMBER_OF_LAYERS})'
    )
    
    args = parser.parse_args()
    
    run_simulation_with_logging(
        num_layers=args.layers,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
