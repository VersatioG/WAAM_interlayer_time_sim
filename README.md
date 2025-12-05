# WAAM Interlayer Time Simulation

This project simulates the thermal behavior during Wire Arc Additive Manufacturing (WAAM) processes. It models temperature evolution in a welding table, base plate, and deposited layers, focusing on interlayer cooling times to optimize manufacturing parameters.

## Features

- Thermal simulation using finite difference methods
- Two simulation modes: fixed wait time after first layer or dynamic waiting until target temperature
- Heat transfer modeling including conduction and radiation
- Linear fitting of wait times to estimate robot control parameters
- Visualization of temperature profiles and wait times

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

- Simulation settings (mode, time step)
- WAAM process parameters (number of layers, temperatures, durations)
- Material properties (thermal conductivities, specific heats, densities)
- Geometry (dimensions of base plate and table)
- Robot fitting mode: choose between "linear" or "cubic" for wait time approximation

## Simulation Details

The simulation uses:

- **Euler explicit method** for time-stepping temperature updates
- **Maier-Kelley equation** for temperature-dependent specific heat of WAAM wire and base plate
- **Fourier's law** for heat conduction between components
- **Stefan-Boltzmann law** for radiative heat loss

## Output

- Console output: Number of layers, mode, calculated wait times, and fitted robot parameters (base time and factor for linear, or coefficients a,b,c,d for cubic, with a >= 0 constraint)
- Plots: Temperature vs. time for all components, and wait time per layer with linear or cubic fit

## Limitations

This simulation provides a simplified thermal model for WAAM processes with several assumptions and constraints:

- **1D Model**: The simulation uses a lumped capacitance approach with discrete nodes for the table, base plate, and layers. It does not account for spatial temperature gradients within each component, which may lead to inaccuracies in larger geometries.
- **Simplified Heat Transfer**: Only conduction between components and radiation to the environment are modeled. Natural convection and forced cooling (e.g., fans) are neglected, potentially underestimating heat dissipation.
- **Constant Material Properties**: Thermal conductivity and density are assumed constant, while only specific heat capacity varies with temperature (Maier-Kelley for WAAM wire and base plate). Real materials may have more complex dependencies.
- **Fixed Time Steps**: The simulation uses a fixed time step (DT), which may not be optimal for all phases of the process. Adaptive time stepping could improve accuracy and efficiency.
- **Idealized Geometry**: Contact areas and layer geometries are approximated. Real WAAM beads have variable cross-sections and may not perfectly match the modeled shapes.
- **No Detailed Welding Physics**: The heat input during welding is simplified by instantly setting new layers to melting temperature. Actual arc physics, including energy deposition rates, are not modeled.

**Conservative Interlayer Time Estimation**: Due to these simplifications, the calculated interlayer wait times tend to be overestimated. This conservative approach ensures that the simulated temperatures reach the target interlayer temperature safely, providing a safety margin in real-world applications where exact thermal conditions may vary.

## License

This project is licensed under the terms specified in the LICENSE file.
