# Simulation State Management - HDF5 Logging

## Overview

The WAAM Thermal Simulation now uses HDF5 for robust state management and continuous data logging.
This system allows:
- **Continuous Logging**: Every logging step is saved immediately (no data loss on crash).
- **Resume Capability**: Resume exactly from the start of the last unfinished layer.
- **Detailed Analysis**: The generated `.h5` file contains the full history of ALL nodes for every timestep.
- **Automatic Resume**: Just restart the script to continue where it left off.

## Quick Start

### 1. Enable HDF5 Logging

In `Thermal_Sim.py`, ensure:

```python
LOGGING_MODE = 2              # Enable file logging
LOG_FILE_NAME = "simulation_state.h5"  # Output filename
```

### 2. Run / Resume

Run the simulation:
```bash
python Thermal_Sim.py
```

- **First Run**: Creates `simulation_state.h5` and logs data.
- **Interruption**: If you stop the simulation (Ctrl+C or crash).
- **Restart**: Run `python Thermal_Sim.py` again. It detects the file, checks parameters, removes partial/incomplete layer data, and resumes correctly.

## Data Structure (.h5)

The HDF5 file contains:

- **Datasets**:
    - `time`: Simulation time [s]
    - `temperatures`: Matrix [time x nodes] with Temperature [°C]
    - `active_mask`: Matrix [time x nodes] (1=Active, 0=Inactive)
    - `radiation_areas`: Matrix [time x nodes] (Area in m²)
    - `layer_indices`: Which layer was being processed [0..N]

- **Attributes**:
    - `parameters`: JSON string of simulation input parameters
    - `last_completed_layer`: Index of the last fully finished layer
    - `wait_times`: List of cooling durations per layer

## Extending Simulation

To add more layers to an existing simulation:

1. Edit `Thermal_Sim.py`:
   ```python
   NUMBER_OF_LAYERS = 20  # Increased from e.g. 15
   ```
2. Run `python Thermal_Sim.py`.
3. The system detects the extension, resizes the dataset columns automatically, initializes new nodes, and continues simulation.

## Parameter Safety

If you change critical parameters (Geometry, Process parameters, etc.) that affect the physics or topology, the system will detect a mismatch with the saved file and **FORCE A RESTART** (overwriting the old file) to ensure data consistency.

Only `NUMBER_OF_LAYERS` is allowed to change.

## Visualization

The `.h5` file is standard HDF5. You can load it in Python:

```python
import h5py
import matplotlib.pyplot as plt

with h5py.File('simulation_state.h5', 'r') as f:
    times = f['time'][:]
    temps = f['temperatures'][:]
    
    # Plot temperature of first bead in Layer 0
    plt.plot(times, temps[:, 10]) 
    plt.show()
```
