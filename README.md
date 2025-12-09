# WAAM Interlayer Time Simulation

This project simulates the thermal behavior during Wire Arc Additive Manufacturing (WAAM) processes. It models temperature evolution in a welding table, base plate, and deposited layers, focusing on interlayer cooling times to optimize manufacturing parameters.

## Features

- **Thermal Simulation**: Finite difference methods with Euler explicit time-stepping
- **Two Simulation Modes**: Fixed wait time after first layer or dynamic waiting until target interlayer temperature
- **Detailed Heat Transfer**: Conduction between components, radiation to environment, and arc power input during welding
- **Bead-Level Modeling**: Individual weld tracks (beads) are simulated for the top two layers, accounting for sequential deposition and arc power distribution
- **Arc Power Distribution**: Configurable heat input during welding, distributed to current bead (50-75%), adjacent beads, and underlying components
- **Temperature-Dependent Properties**: Maier-Kelley equation for specific heat of WAAM wire and base plate
- **Flexible Robot Parameter Fitting**: Linear or cubic fitting of interlayer wait times with non-negative constraints
- **Comprehensive Visualization**: Temperature profiles and wait time analysis with fitting curves

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

1. Run the thermal simulation based on the configured parameters
2. Display results in the console, including fitted robot parameters
3. Generate plots showing temperature profiles and wait times

### Configuration

Modify the input parameters at the top of `Thermal_Sim.py` to adjust:

- **Simulation settings**: Mode (fixed/dynamic wait), time step (DT)
- **WAAM process parameters**: Number of layers, layer height, track geometry (width, overlap, number of tracks, length), process speed, temperatures (melting, interlayer), arc power
- **Material properties**: Maier-Kelley coefficients, thermal conductivities, densities, emissivities for WAAM wire, base plate, and table
- **Geometry**: Dimensions of base plate, table, and contact areas
- **Robot fitting mode**: Choose between "linear" or "cubic" for interlayer wait time approximation (with non-negative constraints)

## Detailed Functionality

### Simulation Overview

The code implements a thermal finite difference simulation for WAAM processes. It models heat transfer through conduction, radiation, and arc power input during welding. The simulation tracks temperatures of the welding table, base plate, and deposited layers over time.

### Key Components

- **run_simulation()**: Main function that orchestrates the layer-by-layer welding process. It calculates geometry, initializes temperatures, and loops through layers, welding beads sequentially.
- **update_temperatures_with_two_bead_layers()**: Core thermal update function for the top two layers modeled at bead level. Handles heat balance including conduction, radiation, and arc power distribution.
- **update_temperatures_with_beads()**: Alternative function for single-layer bead modeling (currently not used in main simulation).
- **Helper Functions**: get_cp_waam() and get_cp_bp() for temperature-dependent specific heat using Maier-Kelley equation.

### Arc Power Implementation

During welding, arc power is distributed as follows:

- **With previous bead in layer**: 50% to current bead, 25% to previous bead, 25% to underlying bead/baseplate
- **Without previous bead**: 75% to current bead, 25% to underlying bead/baseplate

This models localized heat input from the welding arc, focusing energy on the active weld pool and adjacent areas.

### Heat Transfer Mechanisms

- **Conduction**: Between table-baseplate, baseplate-layers, and within bead overlap zones using Fourier's law
- **Radiation**: To environment using Stefan-Boltzmann law, accounting for exposed surface areas
- **Arc Power**: Time-dependent heat input during bead welding, distributed as described above

### Bead-Level Modeling

The top two layers are modeled with individual beads to capture:

- Sequential deposition effects
- Overlap geometry and heat accumulation
- Localized arc power distribution

Older layers are consolidated into lumped nodes for computational efficiency.

### Output Analysis

- **Wait Times**: Calculated interlayer cooling times fitted with linear (a + b*i) or cubic (a + b*i + c*i² + d*i³) functions
- **Temperature Profiles**: Time-series plots showing thermal evolution of all components
- **Fitting Parameters**: Used to predict robot wait times for process optimization

## Simulation Details

The simulation uses:

- **Euler explicit method** for time-stepping temperature updates
- **Maier-Kelley equation** for temperature-dependent specific heat of WAAM wire and base plate
- **Fourier's law** for heat conduction between components and within bead overlap zones
- **Stefan-Boltzmann law** for radiative heat loss from all exposed surfaces (sides and top)
- **Bead-level modeling** for the top two layers: Individual weld tracks are simulated with arc power distribution
- **Mass-weighted averaging** for temperature calculations during layer consolidation
- **Layer consolidation** after cooling to maintain computational efficiency for older layers

## Output

- **Console output**: Number of layers, mode, calculated wait times, and fitted robot parameters (base time and factor for linear, or coefficients a,b,c,d for cubic with a >= 0 constraint)
- **Plots**: Temperature vs. time for all components (with bead-level detail for top layers), and wait time per layer with linear or cubic fit

## Limitations

This simulation provides an advanced thermal model for WAAM processes with several assumptions and constraints:

- **Quasi-2D Model for Top Layers**: The top two layers are modeled at bead-level (individual weld tracks), while older layers use a simplified 1D lumped approach. This balances accuracy with computational efficiency.
- **Simplified Arc Physics**: Heat input during welding is modeled as distributed power input rather than detailed plasma physics or wire melting dynamics.
- **1D Heat Transfer Within Components**: Spatial temperature gradients within the table, base plate, and individual beads are not modeled, which may lead to inaccuracies in larger geometries.
- **Simplified Heat Transfer**: Only conduction between components, bead overlap zones, radiation, and arc power are modeled. Natural convection, forced cooling, and complex arc effects are neglected.
- **Constant Material Properties**: Thermal conductivity and density are assumed constant, while only specific heat capacity varies with temperature (Maier-Kelley for WAAM wire and base plate).
- **Fixed Time Steps**: The simulation uses a fixed time step (DT), which may not be optimal for all phases of the process.
- **Idealized Geometry**: Bead geometries, contact areas, and layer shapes are approximated. Real WAAM beads have variable cross-sections and complex overlap patterns.
- **No Detailed Welding Physics**: Heat input during welding uses simplified power distribution. Actual arc physics, including energy deposition rates and wire feed dynamics, are not modeled.

**Conservative Interlayer Time Estimation**: Despite these simplifications, the bead-level modeling and arc power effects provide more realistic thermal predictions than purely lumped models. The calculated interlayer wait times tend to be conservative, ensuring safe temperature control in real-world WAAM applications.

## License

This project is licensed under the terms specified in the LICENSE file.
