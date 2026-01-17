"""
Test script for state management functionality.
Tests:
1. Basic simulation with LOGGING_MODE=1 (no state file)
2. Simulation with LOGGING_MODE=2 (creates state file)
3. Resume from state file
4. Parameter change detection
"""

import sys
import os
import numpy as np

# Modify Thermal_Sim parameters for quick testing
import Thermal_Sim as ts

# Reduce simulation size for testing
original_layers = ts.NUMBER_OF_LAYERS
original_logging_mode = ts.LOGGING_MODE
original_log_file = ts.LOG_FILE_NAME

# Set test parameters
ts.NUMBER_OF_LAYERS = 2  # Only 2 layers for quick testing
ts.N_LAYERS_AS_BEADS = 2
ts.N_LAYERS_WITH_ELEMENTS = 0  # Disable elements for faster testing
ts.LOGGING_MODE = 1  # Start with plot-only mode
ts.LOG_FILE_NAME = "test_state.pkl"

print("="*60)
print("TEST 1: Simulation with LOGGING_MODE=1 (plot only, no state file)")
print("="*60)

# Clean up any existing test file
if os.path.exists(ts.LOG_FILE_NAME):
    os.remove(ts.LOG_FILE_NAME)

try:
    # This should run without creating a state file
    t_data, layers_data, bp_data, table_data, waits = ts.run_simulation()
    print(f"✓ Simulation completed successfully")
    print(f"  - Final time: {t_data[-1]:.1f}s")
    print(f"  - Layers simulated: {len(waits)}")
    
    # Check that no state file was created
    if not os.path.exists(ts.LOG_FILE_NAME):
        print(f"✓ No state file created (as expected)")
    else:
        print(f"✗ State file was created (unexpected for LOGGING_MODE=1)")
        
except Exception as e:
    print(f"✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST 2: Simulation with LOGGING_MODE=2 (creates state file)")
print("="*60)

# Clean up existing test file
if os.path.exists(ts.LOG_FILE_NAME):
    os.remove(ts.LOG_FILE_NAME)

ts.LOGGING_MODE = 2

try:
    t_data, layers_data, bp_data, table_data, waits = ts.run_simulation()
    print(f"✓ Simulation completed successfully")
    print(f"  - Final time: {t_data[-1]:.1f}s")
    print(f"  - Layers simulated: {len(waits)}")
    
    # Check that state file was created
    if os.path.exists(ts.LOG_FILE_NAME):
        print(f"✓ State file created: {ts.LOG_FILE_NAME}")
        file_size = os.path.getsize(ts.LOG_FILE_NAME)
        print(f"  - File size: {file_size / 1024:.1f} KB")
    else:
        print(f"✗ State file was not created")
        
except Exception as e:
    print(f"✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST 3: Resume from state file (extend simulation)")
print("="*60)

# Extend the simulation by adding more layers
ts.NUMBER_OF_LAYERS = 3

try:
    t_data, layers_data, bp_data, table_data, waits = ts.run_simulation()
    print(f"✓ Simulation resumed and extended successfully")
    print(f"  - Final time: {t_data[-1]:.1f}s")
    print(f"  - Total layers simulated: {len(waits)}")
    
    if len(waits) == 3:
        print(f"✓ Correct number of layers (3)")
    else:
        print(f"✗ Unexpected number of layers: {len(waits)}")
        
except Exception as e:
    print(f"✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST 4: Parameter change detection")
print("="*60)

# Change a critical parameter
original_arc_power = ts.ARC_POWER
ts.ARC_POWER = ts.ARC_POWER * 1.1  # Change arc power by 10%

try:
    t_data, layers_data, bp_data, table_data, waits = ts.run_simulation()
    print(f"✓ Simulation started with changed parameters")
    print(f"  - Should have detected parameter mismatch and started new simulation")
    print(f"  - Total layers simulated: {len(waits)}")
    
    if len(waits) == 3:
        print(f"✓ Simulation completed with new parameters")
    
except Exception as e:
    print(f"✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()

# Restore original parameter
ts.ARC_POWER = original_arc_power

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("All basic tests completed. Check results above for details.")

# Clean up test file
if os.path.exists(ts.LOG_FILE_NAME):
    os.remove(ts.LOG_FILE_NAME)
    print(f"✓ Cleaned up test file: {ts.LOG_FILE_NAME}")

# Restore original values
ts.NUMBER_OF_LAYERS = original_layers
ts.LOGGING_MODE = original_logging_mode
ts.LOG_FILE_NAME = original_log_file

print("\nNote: Full validation requires visual inspection of the plots")
print("and checking that resume functionality works correctly in production.")
