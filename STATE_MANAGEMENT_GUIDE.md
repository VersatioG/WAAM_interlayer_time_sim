# Simulation State Management - Usage Guide

## Overview

The WAAM Thermal Simulation now includes built-in state management that allows you to:
- Save simulation progress automatically
- Resume interrupted simulations
- Extend completed simulations with more layers
- Protect against parameter changes

## Quick Start

### 1. Enable State Management

Edit `Thermal_Sim.py` and set:

```python
LOGGING_MODE = 2              # Enable state management
LOG_FILE_NAME = "sim_state.pkl"  # Your state file name
```

### 2. Run Your Simulation

```bash
python Thermal_Sim.py
```

The simulation will:
- Save state after each layer to `sim_state.pkl`
- Display progress with layer completion status
- Create a backup file (`.backup`) before overwriting existing state

### 3. Resume After Interruption

If the simulation is interrupted (power failure, manual stop, crash), simply run again:

```bash
python Thermal_Sim.py
```

The simulation will:
- Detect the existing state file
- Validate parameters haven't changed
- Resume from the last completed layer
- Continue until `NUMBER_OF_LAYERS` is reached

## Common Use Cases

### Case 1: Extending a Completed Simulation

You ran a simulation with 10 layers and now want 15 layers:

1. Open `Thermal_Sim.py`
2. Change: `NUMBER_OF_LAYERS = 15`
3. Keep: `LOGGING_MODE = 2` and same `LOG_FILE_NAME`
4. Run: `python Thermal_Sim.py`

Result: Simulation resumes from layer 10 and adds layers 11-15.

### Case 2: Shortening a Simulation

You started with 20 layers but realize 15 is enough:

1. Open `Thermal_Sim.py`
2. Change: `NUMBER_OF_LAYERS = 15`
3. Keep: `LOGGING_MODE = 2` and same `LOG_FILE_NAME`
4. Run: `python Thermal_Sim.py`

Result: Simulation uses existing 15 layers from state, skips remaining 5.

### Case 3: Running Without State Management

For quick tests or when you don't need resume capability:

1. Set: `LOGGING_MODE = 1`
2. Run: `python Thermal_Sim.py`

Result: Normal simulation, no state file created.

### Case 4: Starting Fresh

To start a new simulation from scratch:

```bash
# Delete or rename existing state file
rm sim_state.pkl
# Or rename it
mv sim_state.pkl sim_state_old.pkl

# Run simulation
python Thermal_Sim.py
```

## Parameter Validation

### What Happens When Parameters Change?

When you run with `LOGGING_MODE = 2`, the simulation checks:

1. **Does state file exist?**
   - No → Start new simulation
   - Yes → Continue to step 2

2. **Do saved parameters match current parameters?**
   - Yes → Resume from saved state
   - No → Show differences, start new simulation

3. **Special exception: NUMBER_OF_LAYERS**
   - Can be changed without invalidating state
   - Used to extend or shorten simulations

### Parameters That Invalidate State

If you change ANY of these, a new simulation starts:

- Simulation settings (DT, LOGGING_FREQUENCY)
- Discretization (N_LAYERS_AS_BEADS, N_LAYERS_WITH_ELEMENTS, N_ELEMENTS_PER_BEAD)
- WAAM parameters (LAYER_HEIGHT, TRACK_WIDTH, PROCESS_SPEED, temperatures, etc.)
- Material properties (RHO_WAAM, LAMBDA_WAAM, etc.)
- Geometry (BP dimensions, TABLE dimensions)
- Any other input parameter in the INPUT BLOCK

**Exception:** `NUMBER_OF_LAYERS` can be changed freely.

## Console Output Examples

### New Simulation with State Management

```
State logging enabled. Checking for existing state file: sim_state.pkl
No existing state file. Starting new simulation.
Starting simulation...
[... simulation output ...]
Simulating layers: 100%|██████████| 10/10 [02:15<00:00, 13.5s/it]
Simulation state saved to sim_state.pkl
```

### Resuming from Saved State

```
State logging enabled. Checking for existing state file: sim_state.pkl
Simulation state loaded from sim_state.pkl
✓ Compatible state file found. Resuming simulation...
  Resuming from layer 5, time 512.3s
Starting simulation...
Resuming from layer 6/10
Simulating layers:  50%|█████     | 5/10 [00:00<?, ?it/s]
[... continues from layer 6 ...]
```

### Extending Simulation

```
State logging enabled. Checking for existing state file: sim_state.pkl
Simulation state loaded from sim_state.pkl
✓ Compatible state file found. Resuming simulation...
  Extending simulation: 10 → 15 layers
  Resuming from layer 10, time 1023.7s
Starting simulation...
Resuming from layer 11/15
[... adds layers 11-15 ...]
```

### Parameter Change Detected

```
State logging enabled. Checking for existing state file: sim_state.pkl
Simulation state loaded from sim_state.pkl

Parameter mismatch detected:
  ARC_POWER: saved=4740.0, current=5000.0
✗ Incompatible state file (parameters changed). Starting new simulation.
  To resume, please revert parameters to match saved state.
Starting simulation...
[... starts fresh simulation ...]
```

## Best Practices

### 1. Use Descriptive File Names

Instead of generic names, use descriptive ones:

```python
LOG_FILE_NAME = "steel_10layers_highpower.pkl"
```

### 2. Keep State Files Organized

Create a directory for state files:

```bash
mkdir states
```

Then use:

```python
LOG_FILE_NAME = "states/simulation_2024_01_17.pkl"
```

### 3. Regular Backups

State files contain all your simulation progress. Back them up:

```bash
cp sim_state.pkl backups/sim_state_2024_01_17.pkl
```

### 4. Document Your Runs

Keep a log file with your simulation parameters:

```bash
echo "Run started: $(date)" >> simulation_log.txt
echo "Layers: 15, ARC_POWER: 4740W" >> simulation_log.txt
```

## Troubleshooting

### Problem: "Parameter mismatch" but I didn't change anything

**Solution:** Check that you're using the same Python environment and that no dynamic calculations in the INPUT BLOCK changed (e.g., calculations based on external files or system state).

### Problem: Simulation seems stuck at "Resuming from layer X"

**Solution:** This is normal. The progress bar starts from the resume point. The simulation is working correctly.

### Problem: Want to see what parameters are saved in state file

**Solution:** The state file is binary (pickle format). To inspect parameters, you can create a small script:

```python
import pickle
with open('sim_state.pkl', 'rb') as f:
    state = pickle.load(f)
    print(state.parameters)
```

### Problem: State file is very large

**Solution:** State files grow with simulation complexity. This is normal. Typical sizes:
- Simple simulation (5 layers, low discretization): ~50-100 KB
- Complex simulation (20 layers, high discretization): ~500 KB - 2 MB

To reduce size, decrease logging frequency or use coarser discretization.

## Security Note

⚠️ **Important:** State files use Python's pickle format. Only load state files from trusted sources (your own simulations). Never load state files from unknown sources as they could execute malicious code.

## Summary

State management makes your simulations more robust and flexible:

- ✓ Automatic save after each layer
- ✓ Recover from interruptions
- ✓ Extend simulations easily
- ✓ Parameter validation for safety
- ✓ Backup creation for data protection

For most users, simply set `LOGGING_MODE = 2` and let the system handle the rest!
