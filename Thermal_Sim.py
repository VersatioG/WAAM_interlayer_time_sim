import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import quad

# =============================================================================
# INPUT BLOCK (Adjust values here)
# =============================================================================

# --- Simulation Settings ---
MODE = 2  # 1: Fixed wait time after 1st layer | 2: Always wait for temperature
DT = 0.02   # Simulation time step [s] (Smaller = more accurate, but slower)
LOGGING_EVERY_N_STEPS = 10  # Log data every N time steps to reduce memory usage and plotting overhead

# --- Discretization Settings (counted from top) ---
# NOTE: Element-level discretization increases computational cost significantly.
# N_LAYERS_AS_BEADS: Top layers where each track is modeled as individual bead (thermal node)
# N_LAYERS_WITH_ELEMENTS: Top layers where beads are further subdivided (finer resolution)
# N_ELEMENTS_PER_BEAD: Subdivision count per bead (only used if N_LAYERS_WITH_ELEMENTS > 0)
# Currently the simulation uses N_LAYERS_AS_BEADS = 2 (current + previous layer as beads)
# Element-level refinement is prepared but not yet fully implemented
N_LAYERS_AS_BEADS = 3       # Number of top layers modeled as individual beads (default: 2)
N_LAYERS_WITH_ELEMENTS = 2  # Number of top layers where beads are subdivided into elements (0 = disabled)
N_ELEMENTS_PER_BEAD = 5     # Number of elements per bead along track length (if enabled)

# --- WAAM Process Parameters ---
NUMBER_OF_LAYERS = 30
LAYER_HEIGHT = 0.0024       # [m] (e.g., 2.4mm)

# Track geometry
TRACK_WIDTH = 0.0043         # [m] Width of a single weld track (bead width)
TRACK_OVERLAP = 0.738        # Venter distance in percent of track width (e.g., 73.8% overlap)
NUMBER_OF_TRACKS = 5        # Number of parallel tracks per layer
TRACK_LENGTH = 0.1         # [m] Length of each track

# Process parameters
PROCESS_SPEED = 0.010        # [m/s] Welding speed (travel speed)
MELTING_TEMP = 1450.0       # [°C] Temperature at which the wire impacts
INTERLAYER_TEMP = 200.0     # [°C] Max. temp of previous layer before starting next
MODE_1_WAIT_TIME = 30.0     # [s] Only for Mode 1: Forced pause after layer 1
ARC_POWER = 2270.0          # [W] Total arc power during welding
WIRE_FEED_RATE = 0.04       # [m/s] Wire feed rate (typical: 0.03-0.08 m/s)
WIRE_DIAMETER = 0.0012      # [m] Wire diameter (e.g., 1.2mm)

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
LAMBDA_WAAM = 45.0         # [W/(m K)] Thermal conductivity
EPSILON_WAAM = 0.6         # Emissivity (solid)
EPSILON_WAAM_LIQUID = 0.3  # Emissivity (liquid/molten)

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
EPSILON_BP = 0.8            # Emissivity

# --- Geometry & Material: Welding Table ---
TABLE_LENGTH = 2.5         # [m]
TABLE_WIDTH = 1.2          # [m]
TABLE_THICKNESS = 0.2      # [m]
CP_TABLE = 460.0           # [J/(kg K)]
RHO_TABLE = 7850.0         # [kg/m^3]
LAMBDA_TABLE = 45.0        # [W/(m K)]
EPSILON_TABLE = 0.7        # Emissivity

# --- Interaction & Environment ---
AMBIENT_TEMP = 25.0        # [°C]
CONTACT_BP_TABLE = BP_LENGTH * BP_WIDTH * 0.9    # [m^2] Area (Factor 0.9 for holes in Table)
ALPHA_CONTACT = 600.0      # [W/(m^2 K)] Heat transfer coefficient contact between Base Plate and Table (Gap Conductance)
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

def get_epsilon_waam(temp_c):
    """
    Returns the emissivity of WAAM material as a function of temperature.
    Uses a step function with lower emissivity above melting temperature
    (liquid metal has lower emissivity than oxidized solid).
    """
    if temp_c >= MELTING_TEMP:
        return EPSILON_WAAM_LIQUID
    return EPSILON_WAAM

def kelvin(temp_c):
    return temp_c + 273.15

def get_cp_waam_vectorized(temps):
    """Vectorized version of get_cp_waam for numpy arrays."""
    temps = np.atleast_1d(temps)
    result = np.zeros_like(temps, dtype=np.float64)
    liquid_mask = temps >= MELTING_TEMP
    solid_mask = ~liquid_mask
    
    result[liquid_mask] = CP_WAAM_LIQUID
    if np.any(solid_mask):
        tk = temps[solid_mask] + 273.15
        result[solid_mask] = CP_WAAM_A + (CP_WAAM_B * tk) + (CP_WAAM_C * tk**-2)
    return result

def get_epsilon_waam_vectorized(temps):
    """Vectorized version of get_epsilon_waam for numpy arrays."""
    temps = np.atleast_1d(temps)
    result = np.where(temps >= MELTING_TEMP, EPSILON_WAAM_LIQUID, EPSILON_WAAM)
    return result

# =============================================================================
# EFFICIENT DATA STRUCTURES FOR SIMULATION
# =============================================================================

class ThermalModel:
    """
    Efficient thermal model using numpy arrays for simulation.
    Stores geometry, mass, and area matrices for vectorized calculations.
    """
    
    def __init__(self, layer_area, side_area_layer, bead_params):
        """
        Initialize thermal model with geometry parameters.
        
        Args:
            layer_area: Top surface area of consolidated layer [m²]
            side_area_layer: Side surface area of consolidated layer [m²]
            bead_params: Dictionary with bead geometry parameters
        """
        self.layer_area = layer_area
        self.side_area_layer = side_area_layer
        self.bead_params = bead_params
        
        # Pre-compute table and base plate properties
        vol_bp = BP_LENGTH * BP_WIDTH * BP_THICKNESS
        self.m_bp = vol_bp * RHO_BP
        vol_table = TABLE_LENGTH * TABLE_WIDTH * TABLE_THICKNESS
        self.m_table = vol_table * RHO_TABLE
        self.m_layer = layer_area * LAYER_HEIGHT * RHO_WAAM
        
        # Table effective radiation area
        self.area_table_rad = (TABLE_LENGTH * TABLE_WIDTH) - CONTACT_BP_TABLE
        
        # Base plate total surface area
        self.area_bp_total = 2 * (BP_LENGTH * BP_WIDTH + 
                                   BP_LENGTH * BP_THICKNESS + 
                                   BP_WIDTH * BP_THICKNESS)
        
        # Pre-computed bead masses (arrays for vectorized access)
        m_first = bead_params['m_bead_first']
        m_subsequent = bead_params['m_bead_subsequent']
        
        # Bead mass array for fast lookup
        self.bead_masses = np.array([m_first] + [m_subsequent] * (NUMBER_OF_TRACKS - 1))
        
        # Bead areas
        self.bead_areas = np.array([bead_params['bead_area_first']] + 
                                    [bead_params['bead_area_subsequent']] * (NUMBER_OF_TRACKS - 1))
        
        # Bead widths for radiation
        self.bead_widths = np.array([TRACK_WIDTH] + 
                                     [TRACK_WIDTH * (1 - TRACK_OVERLAP)] * (NUMBER_OF_TRACKS - 1))
        
        # Pre-compute radiation areas for all possible bead configurations
        # Cache for different num_beads and is_top_layer combinations
        self._rad_area_cache = {}
        for num_beads in range(1, NUMBER_OF_TRACKS + 1):
            self._rad_area_cache[(num_beads, True)] = self._compute_bead_rad_areas(num_beads, True)
            self._rad_area_cache[(num_beads, False)] = self._compute_bead_rad_areas(num_beads, False)
    
    def _compute_bead_rad_areas(self, num_beads, is_top_layer):
        """Internal method to compute radiation areas (called during initialization)."""
        rad_areas = np.zeros(num_beads, dtype=np.float64)
        
        for i_bead in range(num_beads):
            bead_width = self.bead_widths[i_bead] if i_bead < len(self.bead_widths) else self.bead_widths[-1]
            bead_top = self.bead_areas[i_bead] if i_bead < len(self.bead_areas) else self.bead_areas[-1]
            
            rad_area = bead_top if is_top_layer else 0.0
            
            if num_beads == 1:
                rad_area += 2 * TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
            elif i_bead == 0:
                rad_area += TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
            elif i_bead == num_beads - 1:
                rad_area += TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
            else:
                rad_area += 2 * bead_width * LAYER_HEIGHT
            
            rad_areas[i_bead] = rad_area
        
        return rad_areas
    
    def compute_bead_radiation_areas(self, num_beads, is_top_layer=True):
        """
        Get cached radiation areas for beads in a layer.
        Uses pre-computed cache for O(1) lookup.
        
        Args:
            num_beads: Number of beads currently in layer
            is_top_layer: Whether this is the top layer (affects top surface radiation)
        
        Returns:
            numpy array of radiation areas for each bead
        """
        if num_beads == 0:
            return np.array([], dtype=np.float64)
        
        # Use cache if available
        key = (num_beads, is_top_layer)
        if key in self._rad_area_cache:
            return self._rad_area_cache[key]
        
        # Fallback to calculation if not in cache
        return self._compute_bead_rad_areas(num_beads, is_top_layer)


def update_temperatures_vectorized(T_array, model, num_base_nodes, num_prev_beads, 
                                    is_welding=False, arc_power=0.0, current_bead_idx=0):
    """
    Efficient temperature update using numpy operations.
    
    Args:
        T_array: numpy array of current temperatures
        model: ThermalModel instance with precomputed geometry
        num_base_nodes: Number of base nodes (table + bp + consolidated layers)
        num_prev_beads: Number of beads in previous layer
        is_welding: Whether arc power is active
        arc_power: Effective arc power [W]
        current_bead_idx: Index of bead being welded
    
    Returns:
        numpy array of updated temperatures
    """
    num_nodes = len(T_array)
    num_current_beads = num_nodes - num_base_nodes - num_prev_beads
    
    # Initialize heat balance array
    Q_balance = np.zeros(num_nodes, dtype=np.float64)
    
    # --- Arc power during welding ---
    if is_welding and arc_power > 0:
        current_bead_node = num_base_nodes + num_prev_beads + current_bead_idx
        
        if current_bead_idx > 0:
            # Has previous bead: 50% current, 25% previous, 25% below
            Q_balance[current_bead_node] += 0.5 * arc_power
            prev_bead_node = current_bead_node - 1
            Q_balance[prev_bead_node] += 0.25 * arc_power
            if num_prev_beads > 0:
                bead_below = num_base_nodes + current_bead_idx
                Q_balance[bead_below] += 0.25 * arc_power
            else:
                Q_balance[1] += 0.25 * arc_power  # Baseplate
        else:
            # No previous bead: 75% current, 25% below
            Q_balance[current_bead_node] += 0.75 * arc_power
            if num_prev_beads > 0:
                bead_below = num_base_nodes + current_bead_idx
                Q_balance[bead_below] += 0.25 * arc_power
            else:
                Q_balance[1] += 0.25 * arc_power
    
    # --- Radiation ---
    T_K4 = (T_array + 273.15)**4
    T_amb_K4 = (AMBIENT_TEMP + 273.15)**4
    
    # Table radiation
    Q_balance[0] -= EPSILON_TABLE * STEFAN_BOLTZMANN * model.area_table_rad * (T_K4[0] - T_amb_K4)
    
    # Base plate radiation
    num_consolidated = num_base_nodes - 2
    total_bead_area = 0.0
    if num_prev_beads > 0:
        total_bead_area += np.sum(model.bead_areas[:num_prev_beads])
    if num_current_beads > 0:
        total_bead_area += np.sum(model.bead_areas[:num_current_beads])
    covered_area = num_consolidated * model.layer_area + total_bead_area
    eff_bp_area = max(0, model.area_bp_total - covered_area)
    Q_balance[1] -= EPSILON_BP * STEFAN_BOLTZMANN * eff_bp_area * (T_K4[1] - T_amb_K4)
    
    # Consolidated layers radiation (vectorized)
    if num_consolidated > 0:
        layer_indices = np.arange(2, num_base_nodes)
        epsilon_layers = get_epsilon_waam_vectorized(T_array[layer_indices])
        Q_balance[layer_indices] -= epsilon_layers * STEFAN_BOLTZMANN * model.side_area_layer * (T_K4[layer_indices] - T_amb_K4)
    
    # Previous layer beads radiation
    if num_prev_beads > 0:
        prev_indices = np.arange(num_base_nodes, num_base_nodes + num_prev_beads)
        prev_rad_areas = model.compute_bead_radiation_areas(num_prev_beads, is_top_layer=False)
        epsilon_prev = get_epsilon_waam_vectorized(T_array[prev_indices])
        Q_balance[prev_indices] -= epsilon_prev * STEFAN_BOLTZMANN * prev_rad_areas * (T_K4[prev_indices] - T_amb_K4)
    
    # Current layer beads radiation
    if num_current_beads > 0:
        curr_indices = np.arange(num_base_nodes + num_prev_beads, num_nodes)
        curr_rad_areas = model.compute_bead_radiation_areas(num_current_beads, is_top_layer=True)
        epsilon_curr = get_epsilon_waam_vectorized(T_array[curr_indices])
        Q_balance[curr_indices] -= epsilon_curr * STEFAN_BOLTZMANN * curr_rad_areas * (T_K4[curr_indices] - T_amb_K4)
    
    # --- Conduction ---
    # Table <-> Base plate
    q_contact = ALPHA_CONTACT * CONTACT_BP_TABLE * (T_array[1] - T_array[0])
    Q_balance[0] += q_contact
    Q_balance[1] -= q_contact
    
    # Base plate <-> first layer (or first beads)
    if num_consolidated > 0:
        dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
        lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
        q_cond = lam_eff * model.layer_area / dist * (T_array[2] - T_array[1])
        Q_balance[1] += q_cond
        Q_balance[2] -= q_cond
        
        # Between consolidated layers (vectorized)
        if num_consolidated > 1:
            for i in range(2, num_base_nodes - 1):
                q_layer = LAMBDA_WAAM * model.layer_area / LAYER_HEIGHT * (T_array[i+1] - T_array[i])
                Q_balance[i] += q_layer
                Q_balance[i+1] -= q_layer
    
    # Conduction between beads (horizontal) - previous layer
    if num_prev_beads > 1:
        overlap_width = model.bead_params['overlap_width']
        contact_area = model.bead_params['bead_contact_area'] * LAYER_HEIGHT
        prev_indices = np.arange(num_base_nodes, num_base_nodes + num_prev_beads)
        diff_T = np.diff(T_array[prev_indices])
        q_horiz = LAMBDA_WAAM * contact_area / overlap_width * diff_T
        Q_balance[prev_indices[:-1]] += q_horiz
        Q_balance[prev_indices[1:]] -= q_horiz
    
    # Conduction between beads (horizontal) - current layer
    if num_current_beads > 1:
        overlap_width = model.bead_params['overlap_width']
        contact_area = model.bead_params['bead_contact_area'] * LAYER_HEIGHT
        curr_indices = np.arange(num_base_nodes + num_prev_beads, num_base_nodes + num_prev_beads + num_current_beads)
        diff_T = np.diff(T_array[curr_indices])
        q_horiz = LAMBDA_WAAM * contact_area / overlap_width * diff_T
        Q_balance[curr_indices[:-1]] += q_horiz
        Q_balance[curr_indices[1:]] -= q_horiz
    
    # Vertical conduction: consolidated -> prev beads, prev beads -> current beads
    if num_prev_beads > 0:
        # Consolidated or BP -> prev beads
        dist = LAYER_HEIGHT if num_consolidated > 0 else (BP_THICKNESS / 2 + LAYER_HEIGHT / 2)
        lam = LAMBDA_WAAM if num_consolidated > 0 else (LAMBDA_BP + LAMBDA_WAAM) / 2
        base_idx = num_base_nodes - 1 if num_consolidated > 0 else 1
        
        for i_bead in range(num_prev_beads):
            bead_idx = num_base_nodes + i_bead
            area = model.bead_areas[i_bead] if i_bead < len(model.bead_areas) else model.bead_areas[-1]
            q_vert = lam * area / dist * (T_array[bead_idx] - T_array[base_idx])
            Q_balance[base_idx] += q_vert
            Q_balance[bead_idx] -= q_vert
    
    # Prev beads -> current beads (vertical)
    if num_prev_beads > 0 and num_current_beads > 0:
        for i_bead in range(num_current_beads):
            curr_idx = num_base_nodes + num_prev_beads + i_bead
            prev_idx = num_base_nodes + i_bead
            area = model.bead_areas[i_bead] if i_bead < len(model.bead_areas) else model.bead_areas[-1]
            q_vert = LAMBDA_WAAM * area / LAYER_HEIGHT * (T_array[curr_idx] - T_array[prev_idx])
            Q_balance[prev_idx] += q_vert
            Q_balance[curr_idx] -= q_vert
    elif num_current_beads > 0:
        # Current beads directly on consolidated layer or BP
        dist = LAYER_HEIGHT if num_consolidated > 0 else (BP_THICKNESS / 2 + LAYER_HEIGHT / 2)
        lam = LAMBDA_WAAM if num_consolidated > 0 else (LAMBDA_BP + LAMBDA_WAAM) / 2
        base_idx = num_base_nodes - 1 if num_consolidated > 0 else 1
        
        for i_bead in range(num_current_beads):
            curr_idx = num_base_nodes + num_prev_beads + i_bead
            area = model.bead_areas[i_bead] if i_bead < len(model.bead_areas) else model.bead_areas[-1]
            q_vert = lam * area / dist * (T_array[curr_idx] - T_array[base_idx])
            Q_balance[base_idx] += q_vert
            Q_balance[curr_idx] -= q_vert
    
    # --- Temperature update (Euler explicit) ---
    new_T = T_array.copy()
    
    # Table
    new_T[0] += (Q_balance[0] * DT) / (model.m_table * CP_TABLE)
    
    # Base plate
    cp_bp = get_cp_bp(T_array[1])
    new_T[1] += (Q_balance[1] * DT) / (model.m_bp * cp_bp)
    
    # Consolidated layers (vectorized)
    if num_consolidated > 0:
        layer_indices = np.arange(2, num_base_nodes)
        cp_layers = get_cp_waam_vectorized(T_array[layer_indices])
        new_T[layer_indices] += (Q_balance[layer_indices] * DT) / (model.m_layer * cp_layers)
    
    # Previous layer beads
    if num_prev_beads > 0:
        prev_indices = np.arange(num_base_nodes, num_base_nodes + num_prev_beads)
        cp_prev = get_cp_waam_vectorized(T_array[prev_indices])
        masses_prev = model.bead_masses[:num_prev_beads]
        new_T[prev_indices] += (Q_balance[prev_indices] * DT) / (masses_prev * cp_prev)
    
    # Current layer beads
    if num_current_beads > 0:
        curr_indices = np.arange(num_base_nodes + num_prev_beads, num_nodes)
        cp_curr = get_cp_waam_vectorized(T_array[curr_indices])
        masses_curr = model.bead_masses[:num_current_beads]
        new_T[curr_indices] += (Q_balance[curr_indices] * DT) / (masses_curr * cp_curr)
    
    return new_T


def calculate_wire_melting_power(wire_feed_rate, wire_diameter, ambient_temp, melting_temp, rho_wire):
    """
    Calculates the power required to heat and melt the wire from ambient to melting temperature.
    Uses numerical integration of the temperature-dependent specific heat.
    
    Args:
        wire_feed_rate: Wire feed rate [m/s]
        wire_diameter: Wire diameter [m]
        ambient_temp: Ambient temperature [°C]
        melting_temp: Melting temperature [°C]
        rho_wire: Wire density [kg/m³]
    
    Returns:
        Power required to melt wire [W]
    """
    # Wire cross-sectional area
    wire_area = np.pi * (wire_diameter / 2)**2  # [m²]
    
    # Mass flow rate of wire
    mass_flow_rate = wire_area * wire_feed_rate * rho_wire  # [kg/s]
    
    # Function to integrate: cp as function of temperature in Celsius
    def cp_integrand(temp_c):
        return get_cp_waam(temp_c)  # [J/(kg K)]
    
    # Numerical integration of ∫cp dT from ambient_temp to melting_temp
    energy_per_kg, _ = quad(cp_integrand, ambient_temp, melting_temp)  # [J/kg]
    
    # Power = mass_flow_rate * energy_per_kg
    power_wire_melting = mass_flow_rate * energy_per_kg  # [W]
    
    return power_wire_melting

# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation():
    """
    Main simulation function using efficient numpy arrays.
    Uses ThermalModel class for precomputed geometry parameters.
    """
    # --- Input validation ---
    if N_LAYERS_AS_BEADS > NUMBER_OF_LAYERS:
        raise ValueError(f"N_LAYERS_AS_BEADS ({N_LAYERS_AS_BEADS}) must be <= NUMBER_OF_LAYERS ({NUMBER_OF_LAYERS})")
    
    if N_LAYERS_WITH_ELEMENTS > N_LAYERS_AS_BEADS:
        raise ValueError(f"N_LAYERS_WITH_ELEMENTS ({N_LAYERS_WITH_ELEMENTS}) must be <= N_LAYERS_AS_BEADS ({N_LAYERS_AS_BEADS})")
    
    if N_ELEMENTS_PER_BEAD <= 0:
        raise ValueError(f"N_ELEMENTS_PER_BEAD ({N_ELEMENTS_PER_BEAD}) must be > 0")
    
    if DT <= 0:
        raise ValueError(f"DT ({DT}) must be > 0")
    
    # Check DT stability (rough estimate based on thermal diffusivity)
    # Thermal diffusivity alpha = lambda / (rho * cp) ≈ 4e-6 m²/s for steel
    alpha = LAMBDA_WAAM / (RHO_WAAM * CP_WAAM_A)  # Approximate using solid cp
    dx_min = min(LAYER_HEIGHT, TRACK_WIDTH / max(N_ELEMENTS_PER_BEAD, 1))
    dt_max_stable = dx_min**2 / (2 * alpha)
    if DT > dt_max_stable:
        print(f"Warning: DT ({DT:.3f}s) may be too large for stability. Recommended max: {dt_max_stable:.3f}s")
        print("Consider reducing DT for better accuracy and stability.")
    
    # Calculate effective layer geometry
    effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
    layer_area = effective_layer_width * TRACK_LENGTH
    layer_volume = layer_area * LAYER_HEIGHT
    side_area_layer = 2 * (TRACK_LENGTH * LAYER_HEIGHT + effective_layer_width * LAYER_HEIGHT)
    
    # Calculate layer duration
    total_weld_distance = NUMBER_OF_TRACKS * TRACK_LENGTH
    layer_duration = total_weld_distance / PROCESS_SPEED
    
    # --- Bead geometry ---
    bead_width_effective = TRACK_WIDTH * (1 - TRACK_OVERLAP)
    bead_area_first = TRACK_WIDTH * TRACK_LENGTH
    bead_area_subsequent = bead_width_effective * TRACK_LENGTH
    bead_volume_first = bead_area_first * LAYER_HEIGHT
    bead_volume_subsequent = bead_area_subsequent * LAYER_HEIGHT
    m_bead_first = bead_volume_first * RHO_WAAM
    m_bead_subsequent = bead_volume_subsequent * RHO_WAAM
    
    side_area_bead_first = 2 * TRACK_LENGTH * LAYER_HEIGHT + 2 * TRACK_WIDTH * LAYER_HEIGHT
    side_area_bead_subsequent = 2 * TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width_effective * LAYER_HEIGHT
    
    overlap_width = TRACK_WIDTH * TRACK_OVERLAP
    bead_contact_area = overlap_width * TRACK_LENGTH
    bead_duration = TRACK_LENGTH / PROCESS_SPEED
    
    # Initialize masses
    vol_bp = BP_LENGTH * BP_WIDTH * BP_THICKNESS
    m_bp = vol_bp * RHO_BP
    vol_table = TABLE_LENGTH * TABLE_WIDTH * TABLE_THICKNESS
    m_table = vol_table * RHO_TABLE
    m_layer = layer_volume * RHO_WAAM
    
    # Bead parameters dictionary
    bead_params = {
        'bead_area_first': bead_area_first,
        'bead_area_subsequent': bead_area_subsequent,
        'm_bead_first': m_bead_first,
        'm_bead_subsequent': m_bead_subsequent,
        'side_area_bead_first': side_area_bead_first,
        'side_area_bead_subsequent': side_area_bead_subsequent,
        'bead_contact_area': bead_contact_area,
        'bead_duration': bead_duration,
        'overlap_width': overlap_width
    }
    
    # Create thermal model for efficient calculations
    thermal_model = ThermalModel(layer_area, side_area_layer, bead_params)
    
    # Temperature array (start at ambient) - using numpy for efficiency
    temps = np.array([AMBIENT_TEMP, AMBIENT_TEMP], dtype=np.float64)
    
    # Result storage
    time_log = []
    temp_layers_log = []
    temp_bp_log = []
    temp_table_log = []
    wait_times = []
    
    current_time = 0.0
    logging_counter = 0
    
    # Calculate wire melting power
    power_wire_melting = calculate_wire_melting_power(
        WIRE_FEED_RATE, WIRE_DIAMETER, AMBIENT_TEMP, MELTING_TEMP, RHO_WAAM
    )
    effective_arc_power = ARC_POWER - power_wire_melting
    
    if effective_arc_power < 0:
        raise ValueError(f"Arc power ({ARC_POWER:.1f} W) insufficient to melt wire. "
                        f"Wire melting requires {power_wire_melting:.1f} W.")
    
    print(f"Starting simulation (Mode {MODE})...")
    print(f"Layer geometry: {effective_layer_width*1000:.1f}mm x {TRACK_LENGTH*1000:.1f}mm x {LAYER_HEIGHT*1000:.1f}mm")
    print(f"Layer duration: {layer_duration:.1f}s ({NUMBER_OF_TRACKS} tracks at {PROCESS_SPEED*1000:.1f}mm/s)")
    print(f"Total height after {NUMBER_OF_LAYERS} layers: {NUMBER_OF_LAYERS * LAYER_HEIGHT*1000:.1f}mm")
    print(f"Discretization: {N_LAYERS_AS_BEADS} top layers as beads, {N_LAYERS_WITH_ELEMENTS} with elements ({N_ELEMENTS_PER_BEAD}/bead)")
    print(f"Arc Power: Total = {ARC_POWER:.1f} W, Wire Melting = {power_wire_melting:.1f} W, Effective = {effective_arc_power:.1f} W")
    
    # Main simulation loop
    for i_layer in tqdm(range(NUMBER_OF_LAYERS)):
        
        # Determine structure based on layer position
        if i_layer == 0:
            # First layer: [table, bp] + current_beads
            base_temps = temps.copy()
            num_base_nodes = len(base_temps)
            prev_layer_beads = np.array([], dtype=np.float64)
            current_layer_beads = np.array([], dtype=np.float64)
        else:
            # Subsequent layers: Expand previous layer to beads
            base_temps = temps[:-1].copy()
            num_base_nodes = len(base_temps)
            prev_layer_temp = temps[-1]
            prev_layer_beads = np.full(NUMBER_OF_TRACKS, prev_layer_temp, dtype=np.float64)
            current_layer_beads = np.array([], dtype=np.float64)
        
        # 1. Weld each bead sequentially
        for i_bead in range(NUMBER_OF_TRACKS):
            # Add new bead at melting temperature
            current_layer_beads = np.append(current_layer_beads, MELTING_TEMP)
            
            # Create combined temperature array
            combined_temps = np.concatenate([base_temps, prev_layer_beads, current_layer_beads])
            
            # Simulate welding of this bead
            steps_bead = int(bead_duration / DT)
            
            for _ in range(steps_bead):
                combined_temps = update_temperatures_vectorized(
                    combined_temps, thermal_model, num_base_nodes, 
                    len(prev_layer_beads), is_welding=True, 
                    arc_power=effective_arc_power, current_bead_idx=i_bead
                )
                current_time += DT
                
                # Log data
                logging_counter += 1
                if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                    num_prev = len(prev_layer_beads)
                    log_temps = list(combined_temps[:num_base_nodes])
                    if num_prev > 0:
                        log_temps.append(np.max(combined_temps[num_base_nodes:num_base_nodes + num_prev]))
                    if len(current_layer_beads) > 0:
                        log_temps.append(np.max(combined_temps[num_base_nodes + num_prev:]))
                    log_data(current_time, log_temps, time_log, temp_layers_log, temp_bp_log, temp_table_log)
            
            # Update arrays from combined
            num_prev = len(prev_layer_beads)
            if num_prev > 0:
                prev_layer_beads = combined_temps[num_base_nodes:num_base_nodes + num_prev].copy()
            current_layer_beads = combined_temps[num_base_nodes + num_prev:].copy()
            base_temps = combined_temps[:num_base_nodes].copy()
        
        # 2. Cool until interpass temperature reached
        time_start_wait = current_time
        combined_temps = np.concatenate([base_temps, prev_layer_beads, current_layer_beads])
        
        while True:
            num_prev = len(prev_layer_beads)
            current_beads_temps = combined_temps[num_base_nodes + num_prev:]
            t_hottest_bead = np.max(current_beads_temps) if len(current_beads_temps) > 0 else AMBIENT_TEMP
            
            condition_met = t_hottest_bead <= INTERLAYER_TEMP
            
            # Mode 1 special case
            if MODE == 1 and i_layer == 0:
                if current_time - time_start_wait >= MODE_1_WAIT_TIME:
                    break
            elif condition_met:
                break
            
            combined_temps = update_temperatures_vectorized(
                combined_temps, thermal_model, num_base_nodes,
                len(prev_layer_beads), is_welding=False
            )
            current_time += DT
            
            # Log
            logging_counter += 1
            if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                log_temps = list(combined_temps[:num_base_nodes])
                if num_prev > 0:
                    log_temps.append(np.max(combined_temps[num_base_nodes:num_base_nodes + num_prev]))
                log_temps.append(np.max(combined_temps[num_base_nodes + num_prev:]))
                log_data(current_time, log_temps, time_log, temp_layers_log, temp_bp_log, temp_table_log)
            
            # Update
            if num_prev > 0:
                prev_layer_beads = combined_temps[num_base_nodes:num_base_nodes + num_prev].copy()
            current_layer_beads = combined_temps[num_base_nodes + num_prev:].copy()
            base_temps = combined_temps[:num_base_nodes].copy()
        
        # 3. Merge beads back to single node (mass-weighted average)
        total_bead_mass = m_bead_first + (NUMBER_OF_TRACKS - 1) * m_bead_subsequent
        bead_masses = np.array([m_bead_first] + [m_bead_subsequent] * (NUMBER_OF_TRACKS - 1))
        weighted_temp = np.sum(current_layer_beads * bead_masses[:len(current_layer_beads)]) / total_bead_mass
        
        if len(prev_layer_beads) > 0:
            weighted_prev_temp = np.sum(prev_layer_beads * bead_masses[:len(prev_layer_beads)]) / total_bead_mass
            temps = np.append(base_temps, [weighted_prev_temp, weighted_temp])
        else:
            temps = np.append(base_temps, weighted_temp)
        
        wait_times.append(current_time - time_start_wait)

    return time_log, temp_layers_log, temp_bp_log, temp_table_log, wait_times

def update_temperatures_with_two_bead_layers(T, m_t, m_bp, m_l, layer_area, side_area_layer,
                                               num_base_nodes, num_prev_beads, bead_params, is_welding=False, arc_power=0.0, current_bead_idx=0):
    """
    Calculates one time step DT for all nodes including two layers of individual beads.
    T: [table, base plate, consolidated_layers..., prev_layer_beads..., current_layer_beads...]
    num_base_nodes: Number of nodes before beads (table + bp + consolidated layers)
    num_prev_beads: Number of beads in previous layer (0 if first layer)
    bead_params: Dictionary with bead geometry parameters
    current_bead_idx: Index of the bead currently being welded (0-based within current layer)
    """
    new_T = list(T)
    num_nodes = len(T)
    num_current_beads = num_nodes - num_base_nodes - num_prev_beads
    
    # Extract bead parameters
    bead_area_first = bead_params['bead_area_first']
    bead_area_subsequent = bead_params['bead_area_subsequent']
    m_bead_first = bead_params['m_bead_first']
    m_bead_subsequent = bead_params['m_bead_subsequent']
    side_area_bead_first = bead_params['side_area_bead_first']
    side_area_bead_subsequent = bead_params['side_area_bead_subsequent']
    bead_contact_area = bead_params['bead_contact_area']
    
    # Q_dot array initialization
    Q_balance = np.zeros(num_nodes)
    
    # --- Arc power during welding ---
    # Distribute arc power to: current bead, previous bead (if not first), and bead below (or baseplate for first layer)
    if is_welding and arc_power > 0:
        # Determine which nodes receive arc power
        heated_nodes = []
        
        # 1. Current bead (always receives power)
        current_bead_node_idx = num_base_nodes + num_prev_beads + current_bead_idx
        heated_nodes.append(current_bead_node_idx)
        
        # 2. Previous bead in same layer (if not first bead)
        if current_bead_idx > 0:
            prev_bead_node_idx = num_base_nodes + num_prev_beads + current_bead_idx - 1
            heated_nodes.append(prev_bead_node_idx)
        
        # 3. Bead below (in previous layer) or baseplate (for first layer)
        if num_prev_beads > 0:
            # There is a previous layer - heat the corresponding bead below
            bead_below_node_idx = num_base_nodes + current_bead_idx
            heated_nodes.append(bead_below_node_idx)
        else:
            # First layer - heat the baseplate instead
            heated_nodes.append(1)  # Index 1 is baseplate
        
        # Distribute power according to specified percentages
        # 50% to current bead, 25% to previous (if exists), 25% to below
        # If no previous: 75% to current, 25% to below
        if current_bead_idx > 0:
            # Has previous bead: 50% current, 25% previous, 25% below
            Q_balance[current_bead_node_idx] += 0.5 * arc_power
            prev_bead_node_idx = num_base_nodes + num_prev_beads + current_bead_idx - 1
            Q_balance[prev_bead_node_idx] += 0.25 * arc_power
            if num_prev_beads > 0:
                bead_below_node_idx = num_base_nodes + current_bead_idx
                Q_balance[bead_below_node_idx] += 0.25 * arc_power
            else:
                Q_balance[1] += 0.25 * arc_power  # Baseplate
        else:
            # No previous bead: 75% current, 25% below
            Q_balance[current_bead_node_idx] += 0.75 * arc_power
            if num_prev_beads > 0:
                bead_below_node_idx = num_base_nodes + current_bead_idx
                Q_balance[bead_below_node_idx] += 0.25 * arc_power
            else:
                Q_balance[1] += 0.25 * arc_power  # Baseplate
    
    # --- 1. Radiation to environment ---
    # Table
    area_table_eff = (TABLE_LENGTH * TABLE_WIDTH) - CONTACT_BP_TABLE
    q_rad_table = EPSILON_TABLE * STEFAN_BOLTZMANN * area_table_eff * (kelvin(T[0])**4 - kelvin(AMBIENT_TEMP)**4)
    Q_balance[0] -= q_rad_table
    
    # Base plate
    area_bp_total = 2*(BP_LENGTH*BP_WIDTH + BP_LENGTH*BP_THICKNESS + BP_WIDTH*BP_THICKNESS)
    num_consolidated_layers = num_base_nodes - 2
    total_bead_area = (bead_area_first + (num_prev_beads - 1) * bead_area_subsequent if num_prev_beads > 0 else 0)
    total_bead_area += (bead_area_first + (num_current_beads - 1) * bead_area_subsequent if num_current_beads > 0 else 0)
    covered_area = num_consolidated_layers * layer_area + total_bead_area
    effective_bp_rad_area = max(0, area_bp_total - covered_area)
    q_rad_bp = EPSILON_BP * STEFAN_BOLTZMANN * effective_bp_rad_area * (kelvin(T[1])**4 - kelvin(AMBIENT_TEMP)**4)
    Q_balance[1] -= q_rad_bp
    
    # Consolidated layers radiation
    for i in range(2, num_base_nodes):
        epsilon_layer = get_epsilon_waam(T[i])
        q_rad_layer = epsilon_layer * STEFAN_BOLTZMANN * side_area_layer * (kelvin(T[i])**4 - kelvin(AMBIENT_TEMP)**4)
        Q_balance[i] -= q_rad_layer
    
    # Previous layer beads radiation
    # Previous layer: NO top surface (covered by current layer)
    # Only side surfaces that are exposed:
    # - First bead: 1x length*height (outer long side) + 1x width*height (front/back, but one side blocked by next bead)
    # - Last bead: 1x length*height (outer long side) + 1x width*height (front/back, but one side blocked by prev bead)
    # - Inner beads: 2x width*height (front and back, both long sides blocked by adjacent beads)
    for i_bead in range(num_prev_beads):
        bead_idx = num_base_nodes + i_bead
        
        # Determine bead width for this bead
        if i_bead == 0:
            bead_width = TRACK_WIDTH
        else:
            bead_width = TRACK_WIDTH * (1 - TRACK_OVERLAP)
        
        # Calculate radiating area based on position
        rad_area = 0.0
        
        if num_prev_beads == 1:
            # Only one bead: all sides radiate
            rad_area = 2 * TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        elif i_bead == 0:
            # First bead (leftmost): outer long side + front/back (width sides)
            rad_area = TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        elif i_bead == num_prev_beads - 1:
            # Last bead (rightmost): outer long side + front/back (width sides)
            rad_area = TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        else:
            # Inner beads: only front and back (width sides), long sides blocked by neighbors
            rad_area = 2 * bead_width * LAYER_HEIGHT
        
        # No top surface radiation (covered by current layer)
        
        epsilon_bead = get_epsilon_waam(T[bead_idx])
        q_rad_bead = epsilon_bead * STEFAN_BOLTZMANN * rad_area * (kelvin(T[bead_idx])**4 - kelvin(AMBIENT_TEMP)**4)
        Q_balance[bead_idx] -= q_rad_bead
    
    # Current layer beads radiation
    # Current layer (top): ALL beads radiate on top surface
    # Side surfaces:
    # - First bead: 1x length*height (outer long side) + 1x width*height (front/back)
    # - Last bead: 1x length*height (outer long side) + 1x width*height (front/back)
    # - Inner beads: 2x width*height (front and back only)
    for i_bead in range(num_current_beads):
        bead_idx = num_base_nodes + num_prev_beads + i_bead
        
        # Determine bead dimensions
        if i_bead == 0:
            bead_width = TRACK_WIDTH
            bead_top_area = TRACK_WIDTH * TRACK_LENGTH
        else:
            bead_width = TRACK_WIDTH * (1 - TRACK_OVERLAP)
            bead_top_area = bead_width * TRACK_LENGTH
        
        # Top surface radiation (all beads in current layer radiate on top)
        rad_area = bead_top_area
        
        # Side surfaces based on position
        if num_current_beads == 1:
            # Only one bead: all sides radiate
            rad_area += 2 * TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        elif i_bead == 0:
            # First bead: outer long side + front/back
            rad_area += TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        elif i_bead == num_current_beads - 1:
            # Last bead: outer long side + front/back
            rad_area += TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        else:
            # Inner beads: only front and back
            rad_area += 2 * bead_width * LAYER_HEIGHT
        
        epsilon_bead = get_epsilon_waam(T[bead_idx])
        q_rad_bead = epsilon_bead * STEFAN_BOLTZMANN * rad_area * (kelvin(T[bead_idx])**4 - kelvin(AMBIENT_TEMP)**4)
        Q_balance[bead_idx] -= q_rad_bead
    
    # --- 2. Heat conduction ---
    
    # Table <-> Base plate
    q_cont = ALPHA_CONTACT * CONTACT_BP_TABLE * (T[1] - T[0])
    Q_balance[0] += q_cont
    Q_balance[1] -= q_cont
    
    # Base plate <-> Layer 1
    if num_base_nodes > 2:
        dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
        area_contact = layer_area
        lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
        
        q_cond_1 = (lam_eff * area_contact / dist) * (T[2] - T[1])
        Q_balance[1] += q_cond_1
        Q_balance[2] -= q_cond_1
    
    # Between consolidated layers
    for i in range(2, num_base_nodes - 1):
        dist = LAYER_HEIGHT
        area = layer_area
        
        q_cond_lay = (LAMBDA_WAAM * area / dist) * (T[i+1] - T[i])
        Q_balance[i] += q_cond_lay
        Q_balance[i+1] -= q_cond_lay
    
    # Top consolidated layer <-> Previous layer beads (if exist)
    if num_prev_beads > 0 and num_base_nodes > 2:
        dist = LAYER_HEIGHT
        # Contact with first bead
        area_contact = bead_area_first
        
        q_cond = (LAMBDA_WAAM * area_contact / dist) * (T[num_base_nodes] - T[num_base_nodes - 1])
        Q_balance[num_base_nodes - 1] += q_cond
        Q_balance[num_base_nodes] -= q_cond
    elif num_prev_beads > 0 and num_base_nodes == 2:
        # First layer: prev beads on base plate
        dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
        area_contact = bead_area_first
        lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
        
        q_cond = (lam_eff * area_contact / dist) * (T[num_base_nodes] - T[1])
        Q_balance[1] += q_cond
        Q_balance[num_base_nodes] -= q_cond
    
    # Horizontal conduction between prev layer beads
    for i_bead in range(num_prev_beads - 1):
        bead_idx = num_base_nodes + i_bead
        next_bead_idx = bead_idx + 1
        
        dist = bead_params['overlap_width']
        area = bead_contact_area * LAYER_HEIGHT
        
        q_cond_beads = (LAMBDA_WAAM * area / dist) * (T[next_bead_idx] - T[bead_idx])
        Q_balance[bead_idx] += q_cond_beads
        Q_balance[next_bead_idx] -= q_cond_beads
    
    # Subsequent prev beads to layer below
    for i_bead in range(1, num_prev_beads):
        bead_idx = num_base_nodes + i_bead
        
        if num_base_nodes > 2:
            dist = LAYER_HEIGHT
            area_contact = bead_area_subsequent
            
            q_cond_down = (LAMBDA_WAAM * area_contact / dist) * (T[bead_idx] - T[num_base_nodes - 1])
            Q_balance[num_base_nodes - 1] += q_cond_down
            Q_balance[bead_idx] -= q_cond_down
        else:
            dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
            area_contact = bead_area_subsequent
            lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
            
            q_cond_down = (lam_eff * area_contact / dist) * (T[bead_idx] - T[1])
            Q_balance[1] += q_cond_down
            Q_balance[bead_idx] -= q_cond_down
    
    # Current layer beads to prev layer beads (vertical)
    for i_bead in range(num_current_beads):
        bead_idx = num_base_nodes + num_prev_beads + i_bead
        
        if num_prev_beads > 0:
            # Conduct to corresponding prev layer bead
            prev_bead_idx = num_base_nodes + i_bead
            dist = LAYER_HEIGHT
            
            if i_bead == 0:
                area_contact = bead_area_first
            else:
                area_contact = bead_area_subsequent
            
            q_cond_vert = (LAMBDA_WAAM * area_contact / dist) * (T[bead_idx] - T[prev_bead_idx])
            Q_balance[prev_bead_idx] += q_cond_vert
            Q_balance[bead_idx] -= q_cond_vert
        elif num_base_nodes > 2:
            # No prev beads, conduct to top consolidated layer
            dist = LAYER_HEIGHT
            if i_bead == 0:
                area_contact = bead_area_first
            else:
                area_contact = bead_area_subsequent
            
            q_cond_vert = (LAMBDA_WAAM * area_contact / dist) * (T[bead_idx] - T[num_base_nodes - 1])
            Q_balance[num_base_nodes - 1] += q_cond_vert
            Q_balance[bead_idx] -= q_cond_vert
        else:
            # First layer, no prev beads: conduct to base plate
            dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
            if i_bead == 0:
                area_contact = bead_area_first
            else:
                area_contact = bead_area_subsequent
            lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
            
            q_cond_vert = (lam_eff * area_contact / dist) * (T[bead_idx] - T[1])
            Q_balance[1] += q_cond_vert
            Q_balance[bead_idx] -= q_cond_vert
    
    # Horizontal conduction between current layer beads
    for i_bead in range(num_current_beads - 1):
        bead_idx = num_base_nodes + num_prev_beads + i_bead
        next_bead_idx = bead_idx + 1
        
        dist = bead_params['overlap_width']
        area = bead_contact_area * LAYER_HEIGHT
        
        q_cond_beads = (LAMBDA_WAAM * area / dist) * (T[next_bead_idx] - T[bead_idx])
        Q_balance[bead_idx] += q_cond_beads
        Q_balance[next_bead_idx] -= q_cond_beads
    
    # --- 3. Temperature Update ---
    # Table
    new_T[0] += (Q_balance[0] * DT) / (m_t * CP_TABLE)
    # BP
    cp_bp_val = get_cp_bp(T[1])
    new_T[1] += (Q_balance[1] * DT) / (m_bp * cp_bp_val)
    # Consolidated layers
    for i in range(2, num_base_nodes):
        cp_val = get_cp_waam(T[i])
        new_T[i] += (Q_balance[i] * DT) / (m_l * cp_val)
    # Previous layer beads
    for i_bead in range(num_prev_beads):
        bead_idx = num_base_nodes + i_bead
        if i_bead == 0:
            m_bead = m_bead_first
        else:
            m_bead = m_bead_subsequent
        cp_val = get_cp_waam(T[bead_idx])
        new_T[bead_idx] += (Q_balance[bead_idx] * DT) / (m_bead * cp_val)
    # Current layer beads
    for i_bead in range(num_current_beads):
        bead_idx = num_base_nodes + num_prev_beads + i_bead
        if i_bead == 0:
            m_bead = m_bead_first
        else:
            m_bead = m_bead_subsequent
        cp_val = get_cp_waam(T[bead_idx])
        new_T[bead_idx] += (Q_balance[bead_idx] * DT) / (m_bead * cp_val)
    
    return new_T

def update_temperatures_with_beads(T, m_t, m_bp, m_l, layer_area, side_area_layer,
                                    num_base_nodes, bead_params, current_bead_idx):
    """
    Calculates one time step DT for all nodes including individual beads.
    T: List of current temperatures [table, base plate, layer1, ..., layerN-1, bead1, bead2, ...]
    num_base_nodes: Number of nodes before beads (table + bp + consolidated layers)
    bead_params: Dictionary with bead geometry parameters
    current_bead_idx: Index of the bead currently being welded (for determining active beads)
    """
    new_T = list(T)
    num_nodes = len(T)
    num_beads = num_nodes - num_base_nodes
    
    # Extract bead parameters
    bead_area_first = bead_params['bead_area_first']
    bead_area_subsequent = bead_params['bead_area_subsequent']
    m_bead_first = bead_params['m_bead_first']
    m_bead_subsequent = bead_params['m_bead_subsequent']
    side_area_bead_first = bead_params['side_area_bead_first']
    side_area_bead_subsequent = bead_params['side_area_bead_subsequent']
    bead_contact_area = bead_params['bead_contact_area']
    
    # Q_dot array initialization (in Watts)
    Q_balance = np.zeros(num_nodes)
    
    # --- 1. Radiation to environment (Boltzmann) ---
    # Table
    area_table_eff = (TABLE_LENGTH * TABLE_WIDTH) - CONTACT_BP_TABLE
    q_rad_table = EPSILON_TABLE * STEFAN_BOLTZMANN * area_table_eff * (kelvin(T[0])**4 - kelvin(AMBIENT_TEMP)**4)
    Q_balance[0] -= q_rad_table
    
    # Base plate
    area_bp_total = 2*(BP_LENGTH*BP_WIDTH + BP_LENGTH*BP_THICKNESS + BP_WIDTH*BP_THICKNESS)
    # Subtract covered area (consolidated layers + beads)
    num_consolidated_layers = num_base_nodes - 2
    total_bead_area = bead_area_first + (num_beads - 1) * bead_area_subsequent if num_beads > 0 else 0
    covered_area = num_consolidated_layers * layer_area + total_bead_area
    effective_bp_rad_area = max(0, area_bp_total - covered_area)
    q_rad_bp = EPSILON_BP * STEFAN_BOLTZMANN * effective_bp_rad_area * (kelvin(T[1])**4 - kelvin(AMBIENT_TEMP)**4)
    Q_balance[1] -= q_rad_bp
    
    # Consolidated layers radiation (side surfaces only, top is covered by beads/next layer)
    for i in range(2, num_base_nodes):
        epsilon_layer = get_epsilon_waam(T[i])
        q_rad_layer = epsilon_layer * STEFAN_BOLTZMANN * side_area_layer * (kelvin(T[i])**4 - kelvin(AMBIENT_TEMP)**4)
        Q_balance[i] -= q_rad_layer
    
    # Bead radiation (for single bead layer - top layer)
    # All beads radiate on top surface
    # Side surfaces based on position:
    # - First bead: outer long side + front/back
    # - Last bead: outer long side + front/back  
    # - Inner beads: only front and back (long sides blocked by neighbors)
    for i_bead in range(num_beads):
        bead_idx = num_base_nodes + i_bead
        
        # Determine bead dimensions
        if i_bead == 0:
            bead_width = TRACK_WIDTH
            bead_top_area = TRACK_WIDTH * TRACK_LENGTH
        else:
            bead_width = TRACK_WIDTH * (1 - TRACK_OVERLAP)
            bead_top_area = bead_width * TRACK_LENGTH
        
        # Top surface radiation (all beads radiate on top)
        rad_area = bead_top_area
        
        # Side surfaces based on position
        if num_beads == 1:
            # Only one bead: all sides radiate
            rad_area += 2 * TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        elif i_bead == 0:
            # First bead: outer long side + front/back
            rad_area += TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        elif i_bead == num_beads - 1:
            # Last bead: outer long side + front/back
            rad_area += TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
        else:
            # Inner beads: only front and back
            rad_area += 2 * bead_width * LAYER_HEIGHT
        
        epsilon_bead = get_epsilon_waam(T[bead_idx])
        q_rad_bead = epsilon_bead * STEFAN_BOLTZMANN * rad_area * (kelvin(T[bead_idx])**4 - kelvin(AMBIENT_TEMP)**4)
        Q_balance[bead_idx] -= q_rad_bead
    
    # --- 2. Heat conduction ---
    
    # Table <-> Base plate
    q_cont = ALPHA_CONTACT * CONTACT_BP_TABLE * (T[1] - T[0])
    Q_balance[0] += q_cont
    Q_balance[1] -= q_cont
    
    # Base plate <-> Layer 1 (if consolidated layers exist)
    if num_base_nodes > 2:
        dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
        area_contact = layer_area
        lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
        
        q_cond_1 = (lam_eff * area_contact / dist) * (T[2] - T[1])
        Q_balance[1] += q_cond_1
        Q_balance[2] -= q_cond_1
    
    # Layer i <-> Layer i+1 (conduction between consolidated layers)
    for i in range(2, num_base_nodes - 1):
        dist = LAYER_HEIGHT
        area = layer_area
        
        q_cond_lay = (LAMBDA_WAAM * area / dist) * (T[i+1] - T[i])
        Q_balance[i]   += q_cond_lay
        Q_balance[i+1] -= q_cond_lay
    
    # Top consolidated layer <-> First bead (if beads exist and consolidated layers exist)
    if num_beads > 0 and num_base_nodes > 2:
        dist = LAYER_HEIGHT  # Center to center
        # Contact area is the bead's bottom area
        area_contact = bead_area_first
        
        q_cond_to_bead = (LAMBDA_WAAM * area_contact / dist) * (T[num_base_nodes] - T[num_base_nodes - 1])
        Q_balance[num_base_nodes - 1] += q_cond_to_bead
        Q_balance[num_base_nodes] -= q_cond_to_bead
    elif num_beads > 0 and num_base_nodes == 2:
        # First layer: beads directly on base plate
        dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
        area_contact = bead_area_first
        lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
        
        q_cond_to_bead = (lam_eff * area_contact / dist) * (T[num_base_nodes] - T[1])
        Q_balance[1] += q_cond_to_bead
        Q_balance[num_base_nodes] -= q_cond_to_bead
    
    # Bead <-> Adjacent bead (horizontal heat transfer through overlap zone)
    for i_bead in range(num_beads - 1):
        bead_idx = num_base_nodes + i_bead
        next_bead_idx = bead_idx + 1
        
        # Conduction through overlap contact area
        # Distance is approximately the overlap width (center to center in width direction)
        dist = bead_params['overlap_width']
        area = bead_contact_area * LAYER_HEIGHT  # Contact area in height direction
        
        q_cond_beads = (LAMBDA_WAAM * area / dist) * (T[next_bead_idx] - T[bead_idx])
        Q_balance[bead_idx] += q_cond_beads
        Q_balance[next_bead_idx] -= q_cond_beads
    
    # Subsequent beads also conduct to layer below through their bottom
    for i_bead in range(1, num_beads):
        bead_idx = num_base_nodes + i_bead
        
        if num_base_nodes > 2:
            # Conduct to top consolidated layer
            dist = LAYER_HEIGHT
            area_contact = bead_area_subsequent
            
            q_cond_down = (LAMBDA_WAAM * area_contact / dist) * (T[bead_idx] - T[num_base_nodes - 1])
            Q_balance[num_base_nodes - 1] += q_cond_down
            Q_balance[bead_idx] -= q_cond_down
        else:
            # First layer: beads on base plate
            dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
            area_contact = bead_area_subsequent
            lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
            
            q_cond_down = (lam_eff * area_contact / dist) * (T[bead_idx] - T[1])
            Q_balance[1] += q_cond_down
            Q_balance[bead_idx] -= q_cond_down
    
    # --- 3. Temperature Update (Euler explicit) ---
    # Table
    new_T[0] += (Q_balance[0] * DT) / (m_t * CP_TABLE)
    # BP
    cp_bp_val = get_cp_bp(T[1])
    new_T[1] += (Q_balance[1] * DT) / (m_bp * cp_bp_val)
    # Consolidated layers
    for i in range(2, num_base_nodes):
        cp_val = get_cp_waam(T[i])
        new_T[i] += (Q_balance[i] * DT) / (m_l * cp_val)
    # Beads
    for i_bead in range(num_beads):
        bead_idx = num_base_nodes + i_bead
        if i_bead == 0:
            m_bead = m_bead_first
        else:
            m_bead = m_bead_subsequent
        cp_val = get_cp_waam(T[bead_idx])
        new_T[bead_idx] += (Q_balance[bead_idx] * DT) / (m_bead * cp_val)
    
    return new_T

def update_temperatures(T, m_t, m_bp, m_l, layer_area, side_area_layer):
    """
    Calculates one time step DT for all nodes.
    T: List of current temperatures [table, base plate, layer1, layer2, ...]
    Each layer is tracked individually with heat transfer to adjacent layers.
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
    # Subtract the area covered by layers
    covered_area = (num_nodes - 2) * layer_area
    effective_bp_rad_area = area_bp_total - covered_area
    q_rad_bp = EPSILON_BP * STEFAN_BOLTZMANN * effective_bp_rad_area * (kelvin(T[1])**4 - kelvin(AMBIENT_TEMP)**4)
    Q_balance[1] -= q_rad_bp
    
    # Layer radiation
    # Each layer radiates over its side surfaces
    # The top layer additionally radiates over its top surface
    for i in range(2, num_nodes):
        rad_area = side_area_layer
        if i == num_nodes - 1:  # Top layer
            rad_area += layer_area  # Add top surface area
        epsilon_layer = get_epsilon_waam(T[i])
        q_rad_layer = epsilon_layer * STEFAN_BOLTZMANN * rad_area * (kelvin(T[i])**4 - kelvin(AMBIENT_TEMP)**4)
        Q_balance[i] -= q_rad_layer

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
        # Area is contact area of layer
        area_contact = layer_area
        # Mean conductivity (harmonic mean or simply mean)
        lam_eff = (LAMBDA_BP + LAMBDA_WAAM) / 2
        
        q_cond_1 = (lam_eff * area_contact / dist) * (T[2] - T[1])
        Q_balance[1] += q_cond_1
        Q_balance[2] -= q_cond_1
        
    # Layer i <-> Layer i+1 (conduction between adjacent layers)
    for i in range(2, num_nodes - 1):
        # i is bottom, i+1 is on top
        dist = LAYER_HEIGHT # Distance center to center
        area = layer_area   # Contact area between layers
        
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

def log_data(t, temps, t_log, layers_log, bp_log, table_log):
    t_log.append(t)
    # Store all layer temperatures (indices 2 onwards)
    layer_temps = temps[2:] if len(temps) > 2 else []
    layers_log.append(layer_temps)
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
    t_data, layers_data, bp_data, table_data, waits = run_simulation()
    
    # Extract top layer temperatures for plotting
    top_data = [layers[-1] if layers else AMBIENT_TEMP for layers in layers_data]
    
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
    # Plot all layers with color gradient
    max_layers = max(len(layers) for layers in layers_data)
    colors = plt.cm.Reds(np.linspace(0.3, 1, max_layers))
    
    # Plot each layer's temperature evolution
    for layer_idx in range(max_layers):
        layer_temps_over_time = [layers[layer_idx] if layer_idx < len(layers) else np.nan 
                                  for layers in layers_data]
        # Only label every 5th layer to avoid clutter
        label = f'Layer {layer_idx+1}' if (layer_idx % 5 == 0 or layer_idx == max_layers-1) else None
        plt.plot(t_data, layer_temps_over_time, color=colors[layer_idx], 
                alpha=0.6, linewidth=0.8, label=label)
    
    plt.plot(t_data, bp_data, label='Base plate', color='blue', linewidth=2)
    plt.plot(t_data, table_data, label='Welding table', color='grey', linewidth=2)
    plt.axhline(y=INTERLAYER_TEMP, color='green', linestyle='--', linewidth=2, label='Interpass Temp')
    plt.title('Temperature profile during the process (all layers tracked)')
    plt.ylabel('Temperature [°C]')
    plt.xlabel('Time [s]')
    plt.legend(loc='best', fontsize=8)
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