#!/usr/bin/env python3
"""
Baseline test runner for WAAM simulation optimization.
Records cooling times and execution time for comparison.
"""
import time
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Import the main simulation
from Thermal_Sim import run_simulation, NUMBER_OF_LAYERS

def run_baseline_test():
    """Run the baseline simulation and save metrics."""
    print("="*60)
    print("BASELINE TEST - WAAM INTERLAYER TIME SIMULATION")
    print("="*60)
    print(f"Number of layers: {NUMBER_OF_LAYERS}")
    print("Starting baseline run...")
    print()
    
    # Record start time
    start_time = time.time()
    
    # Run simulation
    t_data, layers_data, bp_data, table_data, wait_times = run_simulation()
    
    # Record end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Prepare results
    results = {
        'execution_time_seconds': execution_time,
        'number_of_layers': len(wait_times),
        'wait_times': wait_times,
        'total_simulation_time': t_data[-1] if t_data else 0,
        'final_temperatures': {
            'baseplate': bp_data[-1] if bp_data else 0,
            'table': table_data[-1] if table_data else 0,
            'top_layer': layers_data[-1][-1] if layers_data and layers_data[-1] else 0
        }
    }
    
    # Save to file
    output_file = 'baseline_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print("="*60)
    print("BASELINE TEST COMPLETED")
    print("="*60)
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Number of layers: {results['number_of_layers']}")
    print(f"Total simulation time: {results['total_simulation_time']:.2f} s")
    print()
    print("Wait times per layer:")
    for i, wt in enumerate(wait_times):
        print(f"  Layer {i+1}: {wt:.4f} s")
    print()
    print(f"Results saved to: {output_file}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    # Remove existing state file to start fresh
    if os.path.exists('simulation_state.h5'):
        os.remove('simulation_state.h5')
        print("Removed existing state file for clean baseline run.")
    
    results = run_baseline_test()
