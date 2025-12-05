import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

# =============================================================================
# INPUT BLOCK (Adjust values here)
# =============================================================================

# --- Simulation Settings ---
MODE = 2  # 1: Fixed wait time after 1st layer | 2: Always wait for temperature
DT = 0.05   # Simulation time step [s] (Smaller = more accurate, but slower)

# --- WAAM Process Parameters ---
NUMBER_OF_LAYERS = 20
LAYER_HEIGHT = 0.0024       # [m] (e.g., 2mm)
LAYER_AREA = 0.0017         # [m^2] (Cross-sectional area of bead * length)
LAYER_DURATION = 15.0       # [s] Welding duration per layer (Arc On Time)
MELTING_TEMP = 1450.0       # [°C] Temperature at which the wire impacts
INTERLAYER_TEMP = 200.0     # [°C] Max. temp of previous layer before starting next
MODE_1_WAIT_TIME = 30.0     # [s] Only for Mode 1: Forced pause after layer 1

# --- Robot Logic ---
# Robot_Wait = Base_Time + (Layer_Index * Factor)  # Linear
# Or cubic: Robot_Wait = a + b*i + c*i^2 + d*i^3
ROBOT_FIT_MODE = "cubic"  # "linear" or "cubic"

# --- Geometry & Material: WAAM Wire ---
# Maier-Kelley coefficients for solid steel (Ex: Structural steel/Stainless steel mix)
# cp = A + B*T + C*T^(-2)  [J/(kg K)]
CP_WAAM_A = 450.0
CP_WAAM_B = 0.28
CP_WAAM_C = -2.0e5
CP_WAAM_LIQUID = 800.0     # [J/(kg K)] Constant capacity when molten
RHO_WAAM = 7800.0          # [kg/m^3] Density
LAMBDA_WAAM = 30.0         # [W/(m K)] Thermal conductivity
EPSILON_WAAM = 0.75        # Emissivity

# --- Geometry & Material: Base Plate ---
BP_LENGTH = 0.15           # [m]
BP_WIDTH = 0.15            # [m]
BP_THICKNESS = 0.01        # [m]
# Maier-Kelley coefficients for base plate
CP_BP_A = 450.0
CP_BP_B = 0.28
CP_BP_C = -2.0e5
CP_BP_LIQUID = 800.0        # [J/(kg K)] Constant capacity when molten
RHO_BP = 7850.0             # [kg/m^3]
LAMBDA_BP = 45.0            # [W/(m K)]
EPSILON_BP = 0.7            # Emissivity

# --- Geometry & Material: Welding Table ---
TABLE_LENGTH = 2.5         # [m]
TABLE_WIDTH = 1.2          # [m]
TABLE_THICKNESS = 0.2      # [m]
CP_TABLE = 460.0           # [J/(kg K)]
RHO_TABLE = 7850.0         # [kg/m^3]
LAMBDA_TABLE = 45.0        # [W/(m K)]
EPSILON_TABLE = 0.6        # Emissivity

# --- Interaction & Environment ---
AMBIENT_TEMP = 25.0        # [°C]
CONTACT_BP_TABLE = BP_LENGTH * BP_WIDTH * 0.75    # [m^2] Area
ALPHA_CONTACT = 200.0      # [W/(m^2 K)] Heat transfer coefficient contact between Base Plate and Table (Gap Conductance)
STEFAN_BOLTZMANN = 5.67e-8 # [W/(m^2 K^4)]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cp_waam(temp_c):
    """
    Calculates the specific heat capacity according to Maier-Kelley.
    With step function at melting temperature.
    """
    if temp_c >= MELTING_TEMP:
        return CP_WAAM_LIQUID
    
    # Maier-Kelley: T in Kelvin for the formula is common, 
    # here simplified on Celsius inputs or we convert.
    # Common MK formulas use Kelvin.
    tk = temp_c + 273.15
    cp = CP_WAAM_A + (CP_WAAM_B * tk) + (CP_WAAM_C * tk**-2)
    return cp

def get_cp_bp(temp_c):
    """
    Calculates the specific heat capacity for the base plate according to Maier-Kelley.
    With step function at melting temperature.
    """
    if temp_c >= MELTING_TEMP:
        return CP_BP_LIQUID
    
    tk = temp_c + 273.15
    cp = CP_BP_A + (CP_BP_B * tk) + (CP_BP_C * tk**-2)
    return cp

def kelvin(temp_c):
    return temp_c + 273.15

# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation():
    # Initialize masses
    vol_bp = BP_LENGTH * BP_WIDTH * BP_THICKNESS
    m_bp = vol_bp * RHO_BP
    
    vol_table = TABLE_LENGTH * TABLE_WIDTH * TABLE_THICKNESS
    m_table = vol_table * RHO_TABLE
    
    vol_layer = LAYER_AREA * LAYER_HEIGHT
    m_layer = vol_layer * RHO_WAAM
    
    # Temperature arrays (start at ambient temp)
    # nodes[0] = table, nodes[1] = base plate, nodes[2...] = layers
    temps = [AMBIENT_TEMP, AMBIENT_TEMP] 
    
    # Result storage
    time_log = []
    temp_top_log = []
    temp_bp_log = []
    temp_table_log = []
    
    wait_times = [] # Actual required wait times per layer
    
    current_time = 0.0
    
    print(f"Starting simulation (Mode {MODE})...")
    
    # Loop over all layers
    for i_layer in tqdm(range(NUMBER_OF_LAYERS)):
        
        # 1. Add new layer
        # The new layer starts immediately with melting temperature
        temps.append(MELTING_TEMP)
        layer_idx = 2 + i_layer # Index in the temps list
        
        # 2. Welding process (Heat Input Phase + Cooling during weld)
        # We simulate for LAYER_DURATION
        steps_weld = int(LAYER_DURATION / DT)
        
        for _ in range(steps_weld):
            temps = update_temperatures(temps, m_table, m_bp, m_layer)
            current_time += DT
            log_data(current_time, temps, time_log, temp_top_log, temp_bp_log, temp_table_log)
            
        # 3. Cooling phase (Wait until interpass reached)
        time_start_wait = current_time
        
        # Check condition
        while True:
            t_top = temps[-1] # Topmost layer
            
            # Termination condition: Temperature low enough?
            condition_met = t_top <= INTERLAYER_TEMP
            
            # Special case Mode 1: First pause (after layer 1, so i_layer=0) fixed predetermined
            if MODE == 1 and i_layer == 0:
                elapsed_wait = current_time - time_start_wait
                if elapsed_wait >= MODE_1_WAIT_TIME:
                    break
            else:
                if condition_met:
                    break
            
            # Compute step
            temps = update_temperatures(temps, m_table, m_bp, m_layer)
            current_time += DT
            log_data(current_time, temps, time_log, temp_top_log, temp_bp_log, temp_table_log)
            
        # Store wait time
        actual_wait = current_time - time_start_wait
        wait_times.append(actual_wait)

    return time_log, temp_top_log, temp_bp_log, temp_table_log, wait_times

def update_temperatures(T, m_t, m_bp, m_l):
    """
    Calculates one time step DT for all nodes.
    T: List of current temperatures [table, base plate, layer1, layer2, ...]
    """
    new_T = T[:]
    num_nodes = len(T)
    
    # Q_dot array initialization (in Watts)
    # Positive = energy in, Negative = energy out
    Q_balance = np.zeros(num_nodes)
    
    # --- 1. Radiation to environment (Boltzmann) ---
    # Table (effective surface simplified)
    area_table_eff = (TABLE_LENGTH * TABLE_WIDTH) - CONTACT_BP_TABLE
    q_rad_table = EPSILON_TABLE * STEFAN_BOLTZMANN * area_table_eff * (kelvin(T[0])**4 - kelvin(AMBIENT_TEMP)**4)
    Q_balance[0] -= q_rad_table
    
    # Base plate (sides + top minus contact area to layer 1)
    # Simplification: We take effective free area
    area_bp_total = 2*(BP_LENGTH*BP_WIDTH + BP_LENGTH*BP_THICKNESS + BP_WIDTH*BP_THICKNESS)
    # If layers are on it, a part is covered. We neglect the small coverage by layers for BP radiation
    q_rad_bp = EPSILON_BP * STEFAN_BOLTZMANN * area_bp_total * (kelvin(T[1])**4 - kelvin(AMBIENT_TEMP)**4)
    Q_balance[1] -= q_rad_bp
    
    # Layer radiation
    # Each layer radiates over its mantle surface. The top one also over top surface.
    # Simplified: Mantle surface per layer
    mantle_layer = 2 * (np.sqrt(LAYER_AREA) * LAYER_HEIGHT) # Very rough approx. of circumference * height
    # We take user input "LAYER_AREA" and "LAYER_HEIGHT". Width = area / height.
    bead_width = LAYER_AREA / LAYER_HEIGHT
    # Length of bead is not explicitly given as number, but volume implicitly.
    # Assumption for simulation: User means the surface of the layer that radiates.
    area_layer_rad = LAYER_AREA # Top surface
    
    # Only the topmost layer radiates fully upwards, the sides radiate always.
    for i in range(2, num_nodes):
        # Radiation only if hotter than environment
        q_rad = EPSILON_WAAM * STEFAN_BOLTZMANN * area_layer_rad * (kelvin(T[i])**4 - kelvin(AMBIENT_TEMP)**4)
        Q_balance[i] -= q_rad

    # --- 2. Heat conduction (conduction) ---
    
    # Table <-> Base plate
    # Q = alpha * A * (T_hot - T_cold)
    q_cont = ALPHA_CONTACT * CONTACT_BP_TABLE * (T[1] - T[0])
    Q_balance[0] += q_cont # Table gets
    Q_balance[1] -= q_cont # BP loses
    
    # Base plate <-> Layer 1 (Index 2)
    if num_nodes > 2:
        # Thermal resistance: R = L / (lambda * A)
        # Distance approx half plate thickness + half layer height
        dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
        # Area is contact area of bead
        area_contact = LAYER_AREA 
        # Mean conductivity (harmonic mean or simply mean)
        lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
        
        q_cond_1 = (lam_eff * area_contact / dist) * (T[2] - T[1])
        Q_balance[1] += q_cond_1
        Q_balance[2] -= q_cond_1
        
    # Layer i <-> Layer i+1
    for i in range(2, num_nodes - 1):
        # i is bottom, i+1 is on top
        dist = LAYER_HEIGHT # Distance center to center
        area = LAYER_AREA
        
        q_cond_lay = (LAMBDA_WAAM * area / dist) * (T[i+1] - T[i])
        Q_balance[i]   += q_cond_lay
        Q_balance[i+1] -= q_cond_lay
        
    # --- 3. Temperature Update (Euler explicit) ---
    # dT = (Q_bal * dt) / (m * cp)
    
    # Table
    new_T[0] += (Q_balance[0] * DT) / (m_t * CP_TABLE)
    # BP
    cp_bp_val = get_cp_bp(T[1])
    new_T[1] += (Q_balance[1] * DT) / (m_bp * cp_bp_val)
    # Layers
    for i in range(2, num_nodes):
        cp_val = get_cp_waam(T[i])
        new_T[i] += (Q_balance[i] * DT) / (m_l * cp_val)
        
    return new_T

def log_data(t, temps, t_log, top_log, bp_log, table_log):
    t_log.append(t)
    top_log.append(temps[-1]) # Always the currently topmost
    bp_log.append(temps[1])
    table_log.append(temps[0])

# =============================================================================
# EVALUATION & PLOT
# =============================================================================

def linear_func(x, a, b):
    return a + x * b

def cubic_func(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def main():
    # Run simulation
    t_data, top_data, bp_data, table_data, waits = run_simulation()
    
    # --- Calculation of robot factor ---
    # We have the list 'waits'. We fit a function based on ROBOT_FIT_MODE
    # x-values are the layer indices [0, 1, 2, ...]
    x_vals = np.arange(len(waits))
    y_vals = np.array(waits)
    
    if ROBOT_FIT_MODE == "linear":
        # Fit linear: y = base_time + i * factor
        # Constraint: base_time (a) >= 0
        popt, _ = curve_fit(linear_func, x_vals, y_vals, bounds=([0, -np.inf], [np.inf, np.inf]))
        base_time_opt = popt[0]
        factor_opt = popt[1]
        fit_func = linear_func
        fit_label = f'Robot Fit: {base_time_opt:.1f} + i*{factor_opt:.2f}'
    elif ROBOT_FIT_MODE == "cubic":
        # Fit cubic: y = a + b*i + c*i^2 + d*i^3
        # Constraint: constant term (a) >= 0
        popt, _ = curve_fit(cubic_func, x_vals, y_vals, bounds=([0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
        a_opt, b_opt, c_opt, d_opt = popt
        fit_func = cubic_func
        fit_label = f'Robot Fit: {a_opt:.1f} + {b_opt:.2f}*i + {c_opt:.3f}*i² + {d_opt:.4f}*i³'
    else:
        raise ValueError("Invalid ROBOT_FIT_MODE. Choose 'linear' or 'cubic'.")
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Number of layers: {NUMBER_OF_LAYERS}")
    print(f"Mode: {MODE}")
    print("-" * 20)
    print(f"Calculated wait times (excerpt):")
    for i, w in enumerate(waits):
        print(f"  Layer {i+1} -> Wait time: {w:.2f} s")
    
    print("-" * 20)
    if ROBOT_FIT_MODE == "linear":
        print("ROBOT PARAMETERS (Linear Approximation):")
        print(f"Formula: Wait time = Base time + (i * Factor)")
        print(f" >> Base time: {base_time_opt:.4f} s")
        print(f" >> Factor:    {factor_opt:.4f} s/Layer")
    elif ROBOT_FIT_MODE == "cubic":
        print("ROBOT PARAMETERS (Cubic Approximation):")
        print(f"Formula: Wait time = a + b*i + c*i² + d*i³")
        print(f" >> a: {a_opt:.4f} s")
        print(f" >> b: {b_opt:.4f} s/Layer")
        print(f" >> c: {c_opt:.6f} s/Layer²")
        print(f" >> d: {d_opt:.8f} s/Layer³")
    print("="*40)

    # --- Plots ---
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Temperature profile
    plt.subplot(2, 1, 1)
    plt.plot(t_data, top_data, label='Top layer (WAAM)', color='red')
    plt.plot(t_data, bp_data, label='Base plate', color='blue')
    plt.plot(t_data, table_data, label='Welding table', color='grey')
    plt.axhline(y=INTERLAYER_TEMP, color='green', linestyle='--', label='Interpass Temp')
    plt.title('Temperature profile during the process')
    plt.ylabel('Temperature [°C]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Wait times & Fit
    plt.subplot(2, 1, 2)
    plt.scatter(x_vals + 1, y_vals, color='black', label='Simulated wait times')
    plt.plot(x_vals + 1, fit_func(x_vals, *popt), color='red', linestyle='--', 
             label=fit_label)
    plt.title(f'Wait time per layer ({ROBOT_FIT_MODE.capitalize()} fit)')
    plt.ylabel('Wait time [s]')
    plt.xlabel('Layer number')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()