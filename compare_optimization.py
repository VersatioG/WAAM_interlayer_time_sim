#!/usr/bin/env python3
"""
Test and compare optimized versions against baseline.
"""
import time
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

def load_baseline():
    """Load baseline results."""
    with open('baseline_results.json', 'r') as f:
        return json.load(f)

def run_test(test_name, run_func):
    """Run a test and compare against baseline."""
    print("\n" + "="*80)
    print(f"TEST: {test_name}")
    print("="*80)
    
    # Remove state file
    if os.path.exists('simulation_state.h5'):
        os.remove('simulation_state.h5')
    
    # Run simulation
    start_time = time.time()
    t_data, layers_data, bp_data, table_data, wait_times = run_func()
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Load baseline
    baseline = load_baseline()
    
    # Compare results
    print(f"\nExecution time: {execution_time:.2f} s (baseline: {baseline['execution_time_seconds']:.2f} s)")
    speedup = baseline['execution_time_seconds'] / execution_time
    print(f"Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    
    # Compare wait times
    baseline_waits = baseline['wait_times']
    max_diff = max(abs(w1 - w2) for w1, w2 in zip(wait_times, baseline_waits))
    rel_diff = max(abs((w1 - w2) / w2) for w1, w2 in zip(wait_times, baseline_waits) if w2 != 0)
    
    print(f"\nWait times validation:")
    print(f"  Max absolute difference: {max_diff:.6f} s")
    print(f"  Max relative difference: {rel_diff*100:.6f} %")
    
    if max_diff < 0.01:  # 10ms tolerance
        print("  ✓ PASS: Wait times match baseline!")
    else:
        print("  ✗ FAIL: Wait times differ from baseline!")
        print("\nDifferences:")
        for i, (w1, w2) in enumerate(zip(wait_times, baseline_waits)):
            diff = w1 - w2
            if abs(diff) > 0.001:
                print(f"    Layer {i+1}: {diff:+.4f} s (baseline: {w2:.4f}, current: {w1:.4f})")
    
    # Compare final temperatures
    final_temps = {
        'baseplate': bp_data[-1] if bp_data else 0,
        'table': table_data[-1] if table_data else 0,
        'top_layer': layers_data[-1][-1] if layers_data and layers_data[-1] else 0
    }
    
    print(f"\nFinal temperatures:")
    for key in ['baseplate', 'table', 'top_layer']:
        baseline_temp = baseline['final_temperatures'][key]
        current_temp = final_temps[key]
        diff = current_temp - baseline_temp
        print(f"  {key}: {current_temp:.2f}°C (baseline: {baseline_temp:.2f}°C, diff: {diff:+.2f}°C)")
    
    result = {
        'test_name': test_name,
        'execution_time': execution_time,
        'speedup': speedup,
        'wait_times': wait_times,
        'max_wait_time_diff': max_diff,
        'max_relative_diff': rel_diff,
        'final_temperatures': final_temps
    }
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_optimization.py <test_name>")
        sys.exit(1)
    
    test_name = sys.argv[1]
    
    # Import the appropriate simulation function
    if test_name == "baseline":
        from Thermal_Sim import run_simulation
    elif test_name.startswith("opt"):
        module_name = f"Thermal_Sim_{test_name}"
        module = __import__(module_name)
        run_simulation = module.run_simulation
    else:
        print(f"Unknown test: {test_name}")
        sys.exit(1)
    
    result = run_test(test_name, run_simulation)
    
    # Save result
    output_file = f"result_{test_name}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")
