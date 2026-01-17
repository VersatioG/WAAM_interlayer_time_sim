#!/usr/bin/env python3
"""
Quick optimization test with fewer layers for faster iteration.
"""
import time
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Temporarily reduce layers for quick testing
import Thermal_Sim
original_layers = Thermal_Sim.NUMBER_OF_LAYERS
Thermal_Sim.NUMBER_OF_LAYERS = 5  # Just 5 layers for quick test

def run_quick_test(test_name):
    """Run a quick test with 5 layers."""
    print(f"\n{'='*80}")
    print(f"QUICK TEST: {test_name} (5 layers)")
    print('='*80)
    
    # Remove state file
    if os.path.exists('simulation_state.h5'):
        os.remove('simulation_state.h5')
    
    # Import the appropriate module
    if test_name == "baseline":
        from Thermal_Sim import run_simulation
    else:
        module_name = f"Thermal_Sim_{test_name}"
        module = __import__(module_name)
        # Also patch the NUMBER_OF_LAYERS in the imported module
        module.NUMBER_OF_LAYERS = 5
        run_simulation = module.run_simulation
    
    # Run simulation
    start_time = time.time()
    t_data, layers_data, bp_data, table_data, wait_times = run_simulation()
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nExecution time: {execution_time:.2f} s")
    print(f"Wait times: {[f'{w:.2f}' for w in wait_times]}")
    
    return {
        'execution_time': execution_time,
        'wait_times': wait_times
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <test_name>")
        sys.exit(1)
    
    test_name = sys.argv[1]
    result = run_quick_test(test_name)
    
    # Save result
    with open(f'quick_result_{test_name}.json', 'w') as f:
        json.dump(result, f, indent=2)
