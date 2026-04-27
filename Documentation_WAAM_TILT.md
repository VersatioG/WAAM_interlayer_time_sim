# WAAM Interlayer Time Simulation: Technical Documentation

## 1. Introduction: The General Problem

Wire Arc Additive Manufacturing (WAAM) is a directed energy deposition technology that uses an electric arc to melt metal wire, depositing it layer by layer to build 3D components. While WAAM offers high deposition rates and cost-effective manufacturing for large metal parts, it suffers from a critical thermal challenge: **heat accumulation**.

Because metal is deposited sequentially at very high temperatures, the underlying layers and the base plate absorb and retain significant amounts of heat. If the deposition proceeds too quickly, the accumulated heat causes the part's temperature to exceed safe metallurgical thresholds, leading to:
- structural distortion and residual stresses.
- Poor metallurgical bonding and irregular bead geometry.
- Collapse or melting of previously deposited layers.

To mitigate this, the manufacturing process must include **interlayer waiting times**—calculated pauses that allow the part to cool down to a safe target temperature (`INTERLAYER_TEMP`) before the next layer is deposited. 

This simulation script was developed to digitally predict the temperature evolution across the welding table, base plate, and deposited layers, dynamically determining the optimal interlayer waiting times for any given geometry and process parameters.

---

## 2. Physical and Mathematical Model

To simulate the complex thermal dynamics of WAAM, the codebase translates continuous physical processes into discrete mathematical models.

### 2.1 Finite Difference Method (FDM)
The core of the simulation relies on the Finite Difference Method using an explicit Euler time-stepping algorithm. The temperature of any given discrete element in the simulation at the next time step ($T_{new}$) is calculated based on its current temperature ($T_{old}$), the time step ($\Delta t$), and the net heat flow:

$$T_{new} = T_{old} + \Delta t \cdot \frac{Q_{in} - Q_{out}}{m \cdot c_p}$$

where $Q$ represents heat flow, $m$ is the mass of the element, and $c_p$ is its specific heat capacity.

### 2.2 Heat Transfer Mechanisms
- **Conduction:** Modeled using Fourier's law of heat conduction between adjacent discrete elements (nodes). Heat flows from hotter nodes to colder neighboring nodes based on the material's thermal conductivity.
- **Radiation:** The simulation accounts for heat lost to the environment from the boundary nodes. Radiation is calculated using the Stefan-Boltzmann law, which becomes highly dominant at the extreme temperatures characteristic of the welding arc.
- **Strategic Shortcut (No Convection):** To prioritize quick simulation times, convection is explicitly omitted from the mathematical model. At the extreme temperatures of the WAAM process, radiative heat loss and internal heat conduction decisively dominate thermal dissipation. Computing fluid dynamics and convective boundary tracking would introduce massive mathematical overhead for negligible accuracy gains on the macro level.

### 2.3 Arc Heat Input Model
The heat provided by the welding arc is modeled by partitioning the total arc power:
1. **Latent and Sensible Heat of the Wire:** A portion of the energy is consumed to melt the incoming wire.
2. **Direct Substrate Heating:** The remaining energy is transferred directly into the printed layer. The script distributes this power spatially among the focal bead, adjacent beads, and underlying components to avoid unphysical temperature spikes in a single mathematical node.

---

## 3. High-Level Architecture

The software is structured into modular components to separate the physics engine, material properties, and data management:

1. **`Thermal_Sim.py` (Core Orchestrator):** Validates geometric constraints, manages the time loop, applies boundary conditions, and calculates the cooling curves and waiting times.
2. **`Material_Properties.py`:** A dedicated module for temperature-dependent thermodynamics, allowing the simulation to accurately model phase changes.
3. **`simulation_state_manager.py` & `async_state_manager.py`:** Handles the persistence of simulation data, allowing runs to be paused, resumed, and logged without blocking the main computational loop.

---

## 4. Detailed Implementation & Architectural Decisions

The transition from a theoretical mathematical model to a highly performant Python script required several key software engineering and algorithmic choices. **The overarching philosophy of this simulation is rapid iteration:** making necessary strategic simplifications and algorithmic shortcuts to heavily reduce simulation times while maintaining acceptable macro-scale accuracy.

### 4.1 Accelerated Simulation via Dynamic Grid Consolidation
**Problem:** A traditional, high-resolution full-3D transient thermal simulation with thousands of nodes modeled over tiny time intervals ($\Delta t$) requires HPC hardware and causes computation times to stretch into days.
**Solution & Strategic Shortcut:** To achieve quick simulation turnaround times, the script employs a **hybrid, quasi-2D lumped capacitance scheme with dynamic layer consolidation**.
- Three distinct discretization levels are used:
    1. **Low-Fidelity Layer-Level Grid:** The base plate and the first few layers are modeled with a coarse grid, treating each layer as a single node. This captures the overall heat accumulation without micro-scale detail.
    2. **Medium-Fidelity Bead-Level Grid:** The next few layers are modeled with moderate resolution, where each "weld bead" is represented as an individual node. This captures the spatial distribution of heat in the critical region just below the active deposition.
    3. **High-Fidelity Node-Level Grid:** The topmost, layers are modeled with even finer resolution, dividing each bead into multiple nodes to capture the thermal gradients and localized heating and reheating effects from the arc.
- The topmost, actively deposited layers (which experience the sharpest thermal gradients) are modeled with high-fidelity, fine resolution at the individual "Node" level.
- As new layers are deposited, the older, underlying layers are moving further away from the active heat source and their internal temperature gradients begin to smooth out. In the simulation, these layers are **aggressively combined and consolidated into coarser grid blocks**.
- **Decision:** By sacrificing micro-scale spatial resolution in the colder, deeper layers where it has minimal impact on the overarching heat accumulation, this dynamic coarsening drastically caps the memory footprint and slashes mathematical operations per time step. This shortcut is the biggest contributor to ensuring the script solves quickly.

### 4.2 Material Thermodynamics (`Material_Properties.py`)
**Problem:** The thermophysical properties of metals (like structural steel S235JR) vary drastically between room temperature, the melting point, and the liquid state. Assuming constant properties yields highly inaccurate cooling curves.
**Solution & Decision:** The codebase utilizes an extensible `Material` class system that calculates properties dynamically based on local temperature:
- **Solid Phase:** Modeled using the rigorous Eurocode 3 standards.
- **Phase Transition:** Implements an *Effective Heat Capacity Method* with Gaussian smoothing to handle the latent heat of fusion. This prevents the mathematical instability (singularities) that occurs when specific heat capacity spikes during melting/solidification.
- **Liquid Phase:** Uses NIST correlations for temperatures above the liquidus point.
- **Decision:** `scipy.interpolate.interp1d` functions were replaced with high-speed `numpy.interp` arrays for property lookups, as profiling revealed interpolation to be a major bottleneck.

### 4.3 Performance Optimization via Numba and Vectorization
**Problem:** Python is an interpreted language; native `for`-loops over thousands of nodes at every fraction of a second are inherently slow.
**Solution & Decision:** 
- **Pre-allocation:** Static NumPy structures (like `NodeMatrix`) are pre-allocated at initialization. This avoids dynamic memory allocation and fragmentation during the simulation ($O(1)$ lookup time for neighbors).
- **JIT Compilation:** The most computationally expensive functions, such as `compute_waam_conduction_numba`, are decorated with `@jit(nopython=True)`. This compiles the Python code directly into optimized machine code via LLVM, bypassing the standard Python interpreter overhead.

### 4.4 Data State Management and Asynchronous I/O
**Problem:** Writing high-fidelity simulation matrices to disk at regular intervals blocked the CPU, causing writing times to dominate the actual simulation times. Furthermore, long simulations are vulnerable to crashes or pauses.
**Solution & Decision:** 
- The codebase uses **HDF5** to store vast arrays of node data compactly.
- **Asynchronous Logging (`async_state_manager.py`):** File I/O was decoupled from the main thread. Simulation state data is dynamically queued and handled by a detached background thread that chunks and writes to disk. 
- **Decision:** Benchmarks showed a ~2.2x speedup on total execution time after moving to async logging. Static grid mappings are only copied intermittently (e.g., every 100 loops) to prevent thread-queue starvation. This architecture also natively allows for **crash-safe recovery**—the simulation can be interrupted and resumed exactly from the last saved layer.

---

## 5. Conclusion & Result Extraction

The ultimate goal of the script is not just to generate heat maps, but to provide actionable data for the robotic control subsystem. 

Instead of merely exporting raw lists of waiting durations, `Thermal_Sim.py` features **predictive extrapolation**. At the end of the simulation, the script automatically applies curve-fitting algorithms (linear, cubic, and logarithmic fits) to the generated cooling times. 

**Decision:** By outputting actual mathematically derived trajectory functions rather than raw data points, the workflow is streamlined. The operator can directly feed the function parameters into the robot's PLC (Programmable Logic Controller) or path-planning software, bridging the gap between theoretical thermal simulation and practical WAAM execution. 
