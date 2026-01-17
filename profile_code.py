#!/usr/bin/env python3
"""
Profile the simulation to identify bottlenecks.
"""
import cProfile
import pstats
import io
import os
import matplotlib
matplotlib.use('Agg')

# Remove state file for clean run
if os.path.exists('simulation_state.h5'):
    os.remove('simulation_state.h5')

from Thermal_Sim import run_simulation

# Run with profiler
pr = cProfile.Profile()
pr.enable()

print("Running simulation with profiler...")
t_data, layers_data, bp_data, table_data, wait_times = run_simulation()

pr.disable()

# Print statistics
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Top 30 functions

print("\n" + "="*80)
print("PROFILING RESULTS - TOP 30 FUNCTIONS BY CUMULATIVE TIME")
print("="*80)
print(s.getvalue())

# Save to file
with open('profile_results.txt', 'w') as f:
    ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
    ps.print_stats(50)
    f.write("\n\n" + "="*80 + "\n")
    f.write("TOP 50 FUNCTIONS BY TOTAL TIME\n")
    f.write("="*80 + "\n")
    ps = pstats.Stats(pr, stream=f).sort_stats('tottime')
    ps.print_stats(50)

print("\nFull results saved to: profile_results.txt")
