# WAAM Interlayer Time Simulation

This project simulates the thermal behavior during Wire Arc Additive Manufacturing (WAAM) processes. WAAM is a metal additive manufacturing technique that builds parts layer-by-layer using an electric arc to melt metal wire, depositing it as weld beads. The simulation focuses on interlayer cooling times—the waiting periods between layer depositions—to optimize manufacturing parameters and prevent defects like distortion or poor bonding.

The code models temperature evolution in the welding setup (table, base plate, and deposited layers) using finite difference methods. It accounts for heat input from the welding arc, heat conduction between components, and radiation to the environment. The top layers are modeled at bead-level (individual weld tracks) for accuracy, while older layers are consolidated for efficiency.

## Features

- **Thermal Simulation**: Finite difference methods with Euler explicit time-stepping
- **Two Simulation Modes**: Fixed wait time after first layer or dynamic waiting until target interlayer temperature
- **Detailed Heat Transfer**: Conduction between components, radiation to environment, and arc power input during welding
- **Bead-Level Modeling**: Individual weld tracks (beads) are simulated for the top layers, accounting for sequential deposition and arc power distribution
- **Arc Power Distribution**: Configurable heat input during welding, distributed to current bead (50-75%), adjacent beads, and underlying components
- **Temperature-Dependent Properties**: Maier-Kelley equation for specific heat of WAAM wire and base plate
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

### Configuration

Modify the input parameters at the top of `Thermal_Sim.py` to adjust:

- **Simulation settings**: Mode (fixed/dynamic wait), time step (DT), discretization levels
- **WAAM process parameters**: Number of layers, layer height, track geometry (width, overlap, number of tracks, length), process speed, temperatures (melting, interlayer), arc power, wire feed rate
- **Material properties**: Maier-Kelley coefficients, thermal conductivities, densities, emissivities for WAAM wire, base plate, and table
- **Geometry**: Dimensions of base plate, table, and contact areas
- **Robot fitting mode**: Choose between "linear" or "cubic" for interlayer wait time approximation (with non-negative constraints)

## How the Simulation Works

### Physical Model

The simulation solves the heat equation using finite differences in a quasi-2D/3D setup:

- **Components**: Welding table (bottom), base plate, and deposited WAAM layers
- **Heat Sources**: Arc power during welding (distributed to beads), wire melting energy (subtracted from arc power)
- **Heat Sinks**: Radiation to environment (Stefan-Boltzmann law), conduction between components (Fourier's law)
- **Material Properties**: Temperature-dependent specific heat (Maier-Kelley equation), constant thermal conductivity and density

### Code Structure

The main simulation is implemented in `Thermal_Sim.py`:

- **Input Block**: Global parameters for easy configuration
- **Helper Functions**: Temperature-dependent properties (e.g., `get_cp_waam()`, `get_epsilon_waam()`)
- **ThermalModel Class**: Precomputes geometry and areas for efficient calculations
- **run_simulation()**: Main loop that builds the part layer-by-layer
  - Calculates bead geometries and masses
  - Welds each bead sequentially with arc power
  - Waits for interlayer temperature between layers
  - Consolidates old layers for efficiency
- **update_temperatures_vectorized()**: Core thermal solver using numpy arrays
  - Computes heat balance (conduction, radiation, arc power)
  - Updates temperatures with Euler explicit method
- **Analysis Functions**: Fits wait times to linear/cubic models for robot programming

### Discretization Levels

The simulation uses a hybrid discretization approach for computational efficiency:

- **Bead-Level (N_LAYERS_AS_BEADS)**: Top layers modeled as individual weld tracks. Captures sequential deposition, overlap heating, and localized arc effects.
- **Element-Level (N_LAYERS_WITH_ELEMENTS)**: Prepared for subdividing beads into smaller elements along the track length. Currently disabled (set to 0) but ready for finer resolution.
- **Consolidated Layers**: Older layers lumped into single thermal nodes.

### Arc Power and Heat Transfer

- **Arc Power Distribution**: During welding, effective power (arc minus wire melting) is distributed:
  - 50% to current bead, 25% to previous bead (if exists), 25% to underlying layer/baseplate
  - Or 75%/25% if no previous bead
- **Wire Melting**: Calculated via numerical integration of temperature-dependent specific heat
- **Conduction**: Between table-baseplate, layers, and bead overlaps
- **Radiation**: From all exposed surfaces, with temperature-dependent emissivity (lower for molten metal)

### Numerical Methods

- **Time Integration**: Euler explicit (simple, stable for small DT)
- **Spatial Discretization**: Lumped nodes for efficiency, with bead-level detail for top layers
- **Stability Checks**: Automatic validation of DT against thermal diffusivity
- **Performance**: Numpy vectorization for fast computation

## Output

- **Console output**: Validation messages, simulation progress, fitted robot parameters
- **Plots**:
  - Temperature vs. time for all components (showing bead-level detail)
  - Wait time per layer with linear/cubic fit

## Limitations

- Quasi-2D model with simplified geometries
- No spatial gradients within components
- Simplified arc physics (no plasma modeling)
- Fixed time steps (may need tuning for stability)
- Conservative interlayer time estimates for safety

## License

This project is licensed under the terms specified in the LICENSE file.
