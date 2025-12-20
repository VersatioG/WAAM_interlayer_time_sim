# WAAM Interlayer Time Simulation

This project simulates the thermal behavior during Wire Arc Additive Manufacturing (WAAM) processes. WAAM is a metal additive manufacturing technique that builds parts layer-by-layer using an electric arc to melt metal wire, depositing it as weld beads. The simulation focuses on interlayer cooling times—the waiting periods between layer depositions—to optimize manufacturing parameters and prevent defects like distortion or poor bonding.

The code models temperature evolution in the welding setup (table, base plate, and deposited layers) using finite difference methods. It accounts for heat input from the welding arc, heat conduction between components, and radiation to the environment. The top layers are modeled at bead-level (individual weld tracks) for accuracy, while older layers are consolidated for efficiency.

## Features

- **Thermal Simulation**: Finite difference methods with Euler explicit time-stepping
- **High-Performance Vectorization**: Optimized using NumPy vectorization for radiation and temperature updates, significantly reducing computation time
- **Dynamic Interlayer Waiting**: Always waits until target interlayer temperature is reached before depositing next layer
- **Detailed Heat Transfer**: Conduction between components, radiation to environment, and arc power input during welding
- **Optimized Node Management**: `NodeMatrix` class with pre-allocated arrays and incremental radiation area tracking for O(1) neighbor lookups
- **Multi-Node Table Model**: Welding table can be discretized as a single node or multi-node 3D grid (Mode 0 = single, Mode 1+ = subdivided)
- **Bead-Level Modeling**: Individual weld tracks (beads) are simulated for the top layers, accounting for sequential deposition and arc power distribution
- **Arc Power Distribution**: Configurable heat input during welding, distributed to current bead (50-75%), adjacent beads, and underlying components
- **Advanced Material Modeling**: Research-based thermodynamic model for S235JR steel (0°C to 2500°C) including:
    - **Eurocode 3 (EN 1993-1-2)** for solid phase (0-1200°C)
    - **Effective Heat Capacity Method** for melting range (Solidus 1517°C - Liquidus 1535°C) with Gaussian smoothing for numerical stability
    - **NIST/Shomate data** for liquid phase (up to 2500°C)
- **Modular Material System**: Extensible `Material` class architecture allowing easy addition of new materials (e.g., Stainless Steel, Inconel)
- **High-Performance Interpolation**: Optimized using `numpy.interp` and vectorized updates for table/WAAM nodes, ensuring minimal overhead from complex material models
- **Flexible Robot Parameter Fitting**: Linear or cubic fitting of interlayer wait times with non-negative constraints
- **Comprehensive Visualization**: Temperature profiles and wait time analysis with fitting curves
- **Efficient Data Structures**: Numpy-based arrays for fast computation, with prepared element-level discretization for finer resolution

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/VersatioG/WAAM_interlayer_time_sim.git
   cd WAAM_interlayer_time_sim
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the simulation script:

```bash
python Thermal_Sim.py
```

The script will:

1. Validate input parameters (e.g., discretization settings, time step stability)
2. Run the thermal simulation based on the configured parameters
3. Display results in the console, including fitted robot parameters
4. Generate plots showing temperature profiles and wait times

### Node Temperature Logging and 3D Visualization

For detailed analysis, you can log all node temperatures at each timestep and visualize them in 3D:

#### Step 1: Run simulation with detailed logging

```bash
python log_node_temperatures.py --output node_temps.h5 --layers 5
```

Options:
- `--output, -o`: Output HDF5 file path (default: `node_temperatures.h5`)
- `--layers, -l`: Number of layers to simulate (default: uses `NUMBER_OF_LAYERS` from config)

#### Step 2: Visualize in 3D

```bash
python visualize_nodes.py --file node_temps.h5
```

Options:
- `--file, -f`: Input HDF5 file (default: `node_temperatures.h5`)
- `--colormap, -c`: Color scheme for temperature (default: `hot`). Options: `hot`, `jet`, `viridis`, `plasma`, etc.
- `--temp-min`: Minimum temperature for color scale (default: auto)
- `--temp-max`: Maximum temperature for color scale (default: auto)
- `--backend`: Visualization library (`plotly`, `matplotlib`, or `auto`)

The visualization shows:
- Each thermal node as a 3D block at its physical position
- Color-coded temperature (blue=cold, red=hot)
- Interactive slider to navigate through timesteps
- Hover tooltips with node details (Plotly backend)
- Play/Pause animation button

### Configuration

Modify the input parameters at the top of `Thermal_Sim.py` to adjust:

- **Simulation settings**: Time step (DT), discretization levels
- **Table discretization**: `TABLE_DISCRETIZATION_MODE` controls table node count (0 = single node, 1 = base 12 nodes, N = base + 3×(N-1) per dimension)
- **WAAM process parameters**: Number of layers, layer height, track geometry (width, overlap, number of tracks, length), process speed, temperatures (melting, interlayer), arc power, wire feed rate
- **Material selection**: Choose from a database of materials (default: S235JR) for WAAM wire, base plate, and table
- **Material properties**: Thermal conductivities, densities, emissivities for WAAM wire, base plate, and table
- **Geometry**: Dimensions of base plate, table, and contact areas
- **Robot fitting mode**: Choose between "linear" or "cubic" for interlayer wait time approximation (with non-negative constraints)

### Adding New Materials

The simulation uses a modular material system defined in `Material_Properties.py`. To add a new material:

1.  Define a new class inheriting from `Material` in `Material_Properties.py`.
2.  Implement the `__init__` method to set physical constants and generate a lookup table.
3.  Register the material in the `_materials` dictionary.
4.  Update the material name in the configuration section of `Thermal_Sim.py`.

Example:
```python
class StainlessSteel316L(Material):
    def __init__(self):
        super().__init__("SS316L")
        # Define properties and lookup table...
```

## How the Simulation Works

### Overview

The WAAM interlayer time simulation models the thermal evolution during the additive manufacturing process using finite difference methods. The simulation builds the part layer-by-layer, tracking temperature changes in the welding table, base plate, and deposited layers. It focuses on calculating optimal interlayer cooling times to ensure each new layer is deposited at the correct temperature for proper bonding and minimal distortion.

The core simulation loop in `run_simulation()` iterates through each layer, depositing weld beads sequentially while accounting for heat input, conduction, and radiation. The model uses a hybrid discretization approach where the top layers are modeled at bead-level resolution for accuracy, while older layers are consolidated into coarser thermal nodes for computational efficiency.

### Detailed Simulation Process

#### 1. Initialization Phase

- **Parameter Validation**: The script first validates critical parameters such as time step stability (DT must satisfy the Fourier number for numerical stability), discretization settings, and physical constraints.
- **Thermal Model Setup**: A `ThermalModel` object is created to precompute geometric properties and contact areas between components (table-baseplate, baseplate-layers, layer-layer interfaces).
- **Node Matrix Initialization**: A `NodeMatrix` object initializes the thermal nodes representing the welding setup using pre-allocated NumPy arrays:
  - Table nodes (bottom component)
  - Base plate node
  - WAAM layer nodes (pre-allocated for maximum capacity)
- **Incremental Radiation Tracking**: Radiation areas are tracked incrementally as nodes are activated or consolidated, avoiding expensive neighbor searches during the simulation loop.
- **Material Properties**: Temperature-dependent properties are implemented using a research-based thermodynamic model for S235JR (Eurocode 3 + Effective Heat Capacity Method + NIST data), optimized with vectorized lookup tables.

#### 2. Table Discretization

The welding table can be modeled with varying levels of detail using the `TABLE_DISCRETIZATION_MODE` parameter:

- **Mode 0**: Single node representing the entire table
- **Mode 1**: Subdivided grid with 3×2×2 = 12 nodes (configured by `N_TABLE_X`, `N_TABLE_Y`, `N_TABLE_Z`)
- **Mode N (N > 1)**: Extended grid with `(N_TABLE_X + N-1) × (N_TABLE_Y + N-1) × (N_TABLE_Z + N-1)` nodes

Each table node includes:
- **Internal Conduction**: Fourier heat transfer between adjacent nodes in x, y, and z directions
- **Selective Radiation**: Only exterior faces of the grid radiate to the environment, interior faces use internal conduction only
- **Base Plate Contact**: The base plate is positioned at the top-corner node for proper thermal contact
- **Validation**: Ensures table node dimensions (dx, dy) don't exceed base plate dimensions to prevent overlap

#### 3. Layer-by-Layer Deposition Loop

For each layer `i_layer` from 0 to `NUMBER_OF_LAYERS - 1`:

##### 3.1 Layer Geometry Calculation

- Compute the total height of deposited material so far.
- Calculate the number of tracks per layer based on layer width and bead overlap.
- Determine discretization level for the current layer:
  - `current_num_layers = i_layer + 1`
  - `use_elements = (N_LAYERS_WITH_ELEMENTS >= 1)` - Current top layer uses element-level if enabled
  - `use_beads = (N_LAYERS_AS_BEADS >= 1)` - Current top layer uses bead-level if enabled
  - Older layers are consolidated based on distance from top: `layers_from_current_top = current_num_layers - layer_idx - 1`

##### 3.2 Track Deposition Within Layer

For each track `i_track` in the layer:

###### 3.2.1 Bead Geometry and Mass Calculation

- Calculate bead dimensions (width, height, length) based on process parameters.
- Compute bead volume and mass using material density.
- Determine bead position and overlap with adjacent beads.

###### 3.2.2 Sequential Bead Welding

For each bead `i_bead` in the track:

**Welding Phase:**

- **Arc Power Distribution**: During the welding time (bead length / welding speed), effective arc power (total arc power minus wire melting energy) is distributed:
  - `ARC_POWER_CURRENT_FRACTION` (default 0.5) to the current bead
  - Remaining fraction distributed to adjacent beads and underlying components based on area-weighted neighbors
- **Wire Melting Energy**: Calculated by integrating temperature-dependent specific heat from room temperature to melting point, subtracted from total arc power.
- **Temperature Update**: For each time step during welding:
  - Compute heat conduction between all thermal nodes (table ↔ baseplate ↔ layers ↔ beads)
  - Calculate radiation losses for all active nodes using vectorized Stefan-Boltzmann calculations with pre-computed radiation areas
  - Update temperatures for all active nodes simultaneously using vectorized Euler explicit finite difference method: `T_new = T_old + DT * (heat_sources - heat_sinks) / (mass * cp)`

**Inter-Bead Waiting (Mode 1 only):**

- After welding each bead, wait until the interlayer temperature is reached before depositing the next bead.
- This ensures sequential cooling between beads within a track.

##### 3.3 Interlayer Waiting

After completing all tracks in a layer:

- **Dynamic Wait**: Wait until the maximum temperature in the top layer drops to `INTERLAYER_TEMPERATURE`.
- During waiting periods, continue thermal simulation with no arc power input, allowing natural cooling through conduction and radiation.

##### 3.4 Layer Consolidation

After interlayer waiting:

- **Element-to-Bead Consolidation**: If elements exist beyond `N_LAYERS_WITH_ELEMENTS`, combine fine elements into bead-level nodes.
- **Bead-to-Layer Consolidation**: If beads exist beyond `N_LAYERS_AS_BEADS`, combine bead-level nodes into layer-level nodes.
- **Index Recalculation**: Update special node indices (`table_idx`, `bp_idx`) after node deletion to maintain correct thermal connections.

#### 4. Post-Simulation Analysis

- **Wait Time Extraction**: Collect interlayer wait times for each layer.
- **Robot Parameter Fitting**: Fit wait times to a mathematical model:
  - Linear: `wait_time = a + b * layer_index`
  - Cubic: `wait_time = a + b*i + c*i² + d*i³` (with non-negative constraints)
- **Visualization**: Generate plots showing temperature evolution and fitted wait time curves.

### Physical Modeling Details

#### Heat Transfer Mechanisms

- **Conduction**: Fourier's law between contacting surfaces (table-baseplate, layer interfaces, bead overlaps)
- **Radiation**: Stefan-Boltzmann law from all exposed surfaces, with emissivity varying with temperature (lower for molten metal)
- **Arc Heating**: Localized heat input during welding, distributed to current and adjacent thermal nodes

#### Material Properties

- **Specific Heat**: Research-based thermodynamic model (Eurocode 3, Effective Heat Capacity Method, NIST/Shomate) for S235JR steel.
- **Thermal Conductivity**: Constant values for each material
- **Density**: Constant for mass calculations
- **Emissivity**: Temperature-dependent for radiation calculations

#### Numerical Methods

- **Time Integration**: Euler explicit method for simplicity and stability with small time steps
- **Spatial Discretization**: Lumped capacitance model with hybrid resolution (bead/element level for top layers, consolidated for older layers)
- **Stability**: Automatic DT validation against thermal diffusivity limits
- **Performance**: Numpy vectorization for efficient computation of large thermal networks

### Key Features of the Dynamic Discretization

The simulation employs a dynamic discretization strategy that optimizes computational efficiency while maintaining accuracy:

- **Current Layer Priority**: The topmost layer always receives the finest discretization level available (elements if `N_LAYERS_WITH_ELEMENTS > 0`, otherwise beads if `N_LAYERS_AS_BEADS > 0`)
- **Progressive Consolidation**: As new layers are added, older layers are automatically consolidated from elements → beads → layers based on their distance from the current top
- **Memory Efficiency**: Prevents exponential growth of thermal nodes by consolidating distant layers
- **Accuracy Preservation**: Maintains detailed thermal gradients in recently deposited material where they matter most for interlayer cooling decisions

This approach ensures that the simulation can handle large builds (30+ layers) while keeping computation times reasonable, with the most critical thermal interactions modeled at appropriate resolution levels.

### Performance and Optimization

The simulation is designed for high performance to allow for rapid parameter studies:
- **Vectorized Operations**: Most thermal calculations (radiation, temperature updates, property evaluations) are performed on entire arrays at once using NumPy, minimizing Python loop overhead.
- **O(1) Neighbor Lookups**: The `NodeMatrix` uses a structured grid mapping to find neighbors in constant time, avoiding expensive search algorithms.
- **Incremental State Updates**: Properties like radiation area and baseplate coverage are updated only when the geometry changes (activation/consolidation), rather than being recalculated every time step.
- **Pre-allocation**: All major data arrays are pre-allocated at startup to avoid memory fragmentation and allocation overhead during the simulation.

## Output

- **Console output**: Validation messages, simulation progress, fitted robot parameters
- **Plots**:
  - Temperature vs. time for all components (showing bead-level detail)
  - Welding table temperature shows maximum (hotspot) rather than average
  - Wait time per layer with linear/cubic fit
- **HDF5 Data Files** (from `log_node_temperatures.py`):
  - Timestep arrays
  - Node metadata (positions, dimensions, types)
  - Full temperature history for all nodes
- **Interactive 3D Visualization** (from `visualize_nodes.py`):
  - HTML file with Plotly visualization (if using Plotly backend)
  - Matplotlib window with slider controls

## Project Structure

```
WAAM_interlayer_time_sim/
├── Thermal_Sim.py           # Main simulation script
├── log_node_temperatures.py  # Detailed node temperature logging
├── visualize_nodes.py        # 3D visualization with timestep slider
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
└── LICENSE                   # License file
```

## Limitations

- Quasi-2D model with simplified geometries
- No spatial gradients within components
- Simplified arc physics (no plasma modeling)
- Fixed time steps (may need tuning for stability)
- Conservative interlayer time estimates for safety

## License

This project is licensed under the terms specified in the LICENSE file.
