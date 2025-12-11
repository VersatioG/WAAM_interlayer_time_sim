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

class NodeMatrix:
    """
    Matrix-based node tracking for layers, beads, and elements.
    Each node has: layer_num, bead_num, element_num, and discretization level.
    Allows easy neighbor finding (±1 indexing) and radiation surface determination.
    """
    
    def __init__(self):
        # Node attributes: layer, bead, element indices, and level type
        self.layer_idx = []      # Layer number (0-based)
        self.bead_idx = []       # Bead number within layer (0-based, -1 for layer-level)
        self.element_idx = []    # Element number within bead (0-based, -1 for bead/layer-level)
        self.level_type = []     # 'layer', 'bead', or 'element'
        
        # Physical properties per node
        self.masses = []         # Mass of each node [kg]
        self.areas = []          # Contact area (for conduction) [m²]
        self.temperatures = []   # Current temperature [°C]
        
        # Special nodes (table, baseplate)
        self.table_idx = None
        self.bp_idx = None
        
    def add_node(self, layer_idx, bead_idx, element_idx, level_type, mass, area, temperature):
        """Add a new node to the matrix."""
        idx = len(self.layer_idx)
        self.layer_idx.append(layer_idx)
        self.bead_idx.append(bead_idx)
        self.element_idx.append(element_idx)
        self.level_type.append(level_type)
        self.masses.append(mass)
        self.areas.append(area)
        self.temperatures.append(temperature)
        return idx
    
    def get_top_layer_idx(self):
        """Get the layer index of the topmost layer."""
        if not self.layer_idx:
            return -1
        waam_nodes = [i for i, lt in enumerate(self.level_type) if lt != 'special']
        if not waam_nodes:
            return -1
        return max(self.layer_idx[i] for i in waam_nodes)
    
    def get_nodes_in_layer(self, layer_idx):
        """Get all node indices in a specific layer."""
        return [i for i in range(len(self.layer_idx)) 
                if self.layer_idx[i] == layer_idx and self.level_type[i] != 'special']
    
    def get_vertical_neighbor(self, node_idx, direction):
        """
        Get vertical neighbor (±1 in layer direction).
        direction: +1 for above, -1 for below
        Returns node index or None if not found.
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        level = self.level_type[node_idx]
        
        target_layer = layer + direction
        
        # Find matching node in target layer
        for i in range(len(self.layer_idx)):
            if (self.layer_idx[i] == target_layer and 
                self.bead_idx[i] == bead and 
                self.element_idx[i] == element and
                self.level_type[i] == level):
                return i
        
        # If exact match not found, try base plate or consolidated layer
        if target_layer < 0:
            return self.bp_idx
        
        return None
    
    def get_horizontal_neighbor(self, node_idx, direction):
        """
        Get horizontal neighbor (±1 in bead direction).
        direction: +1 for next bead, -1 for previous bead
        Returns node index or None if not found.
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        level = self.level_type[node_idx]
        
        if bead == -1:  # Layer-level nodes have no horizontal neighbors
            return None
        
        target_bead = bead + direction
        
        # Find matching node in target bead
        for i in range(len(self.layer_idx)):
            if (self.layer_idx[i] == layer and 
                self.bead_idx[i] == target_bead and 
                self.element_idx[i] == element and
                self.level_type[i] == level):
                return i
        
        return None
    
    def get_longitudinal_neighbor(self, node_idx, direction):
        """
        Get longitudinal neighbor (±1 in element direction along bead).
        direction: +1 for next element, -1 for previous element
        Returns node index or None if not found.
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        level = self.level_type[node_idx]
        
        if element == -1 or level != 'element':
            return None
        
        target_element = element + direction
        
        # Find matching node
        for i in range(len(self.layer_idx)):
            if (self.layer_idx[i] == layer and 
                self.bead_idx[i] == bead and 
                self.element_idx[i] == target_element and
                self.level_type[i] == level):
                return i
        
        return None
    
    def compute_radiation_area(self, node_idx, top_layer_idx):
        """
        Compute radiation area for a node based on its position in the matrix.
        Uses the logic: only top layer radiates with top surface, 
        layer-level radiates full side, bead-level considers position.
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        level = self.level_type[node_idx]
        
        if level == 'special':
            return 0.0
        
        rad_area = 0.0
        is_top = (layer == top_layer_idx)
        
        if level == 'layer':
            # Consolidated layer: side surfaces only
            if is_top:
                rad_area += self.areas[node_idx]  # Top surface
            rad_area += 2 * (TRACK_LENGTH * NUMBER_OF_TRACKS * (1 - TRACK_OVERLAP) + 
                            TRACK_LENGTH) * LAYER_HEIGHT  # Simplified side area
            
        elif level == 'bead':
            # Bead-level consideration
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            
            # Top surface (only if top layer)
            if is_top:
                rad_area += self.areas[node_idx]
            
            # Count beads in this layer
            beads_in_layer = sum(1 for i in range(len(self.layer_idx)) 
                                if self.layer_idx[i] == layer and self.bead_idx[i] >= 0)
            
            # Side surfaces based on position
            if beads_in_layer == 1:
                # Only bead: all sides
                rad_area += 2 * TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
            elif bead == 0 or bead == beads_in_layer - 1:
                # First or last bead: one long side + short sides
                rad_area += TRACK_LENGTH * LAYER_HEIGHT + 2 * bead_width * LAYER_HEIGHT
            else:
                # Inner bead: only short sides
                rad_area += 2 * bead_width * LAYER_HEIGHT
        
        elif level == 'element':
            # Element-level consideration
            element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            
            # Top surface (only if top layer)
            if is_top:
                rad_area += element_length * bead_width
            
            # Count elements and beads
            beads_in_layer = sum(1 for i in range(len(self.layer_idx)) 
                                if self.layer_idx[i] == layer and self.bead_idx[i] >= 0)
            
            # Broad side (perpendicular to track direction)
            if element == 0 or element == N_ELEMENTS_PER_BEAD - 1:
                # First or last element: has broad side exposed
                rad_area += bead_width * LAYER_HEIGHT
            
            # Long sides (parallel to track direction)
            if bead == 0 or bead == beads_in_layer - 1:
                # First or last bead: element has fraction of long side exposed
                rad_area += element_length * LAYER_HEIGHT
        
        return rad_area
    
    def to_arrays(self):
        """Convert to numpy arrays for efficient computation."""
        return {
            'layer_idx': np.array(self.layer_idx, dtype=np.int32),
            'bead_idx': np.array(self.bead_idx, dtype=np.int32),
            'element_idx': np.array(self.element_idx, dtype=np.int32),
            'masses': np.array(self.masses, dtype=np.float64),
            'areas': np.array(self.areas, dtype=np.float64),
            'temperatures': np.array(self.temperatures, dtype=np.float64)
        }


class ThermalModel:
    """
    Thermal model using NodeMatrix for efficient simulation.
    Manages geometry and heat transfer calculations.
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


def update_temperatures_matrix(node_matrix, model, is_welding=False, arc_power=0.0, welding_node_idx=None):
    """
    Update temperatures using NodeMatrix for efficient neighbor finding and radiation calculation.
    
    Args:
        node_matrix: NodeMatrix instance with all nodes
        model: ThermalModel instance with precomputed geometry
        is_welding: Whether arc power is active
        arc_power: Effective arc power [W]
        welding_node_idx: Index of node being welded
    """
    num_nodes = len(node_matrix.temperatures)
    Q_balance = np.zeros(num_nodes, dtype=np.float64)
    
    # Get temperature array
    T = np.array(node_matrix.temperatures, dtype=np.float64)
    
    # --- Arc power during welding ---
    if is_welding and arc_power > 0 and welding_node_idx is not None:
        # Distribute arc power: 50% to current, 25% to horizontal neighbor, 25% to vertical neighbor
        Q_balance[welding_node_idx] += 0.5 * arc_power
        
        # Horizontal neighbor (previous bead in same layer)
        h_neighbor = node_matrix.get_horizontal_neighbor(welding_node_idx, -1)
        if h_neighbor is not None:
            Q_balance[h_neighbor] += 0.25 * arc_power
            # Vertical neighbor gets remaining 25%
            v_neighbor = node_matrix.get_vertical_neighbor(welding_node_idx, -1)
            if v_neighbor is not None:
                Q_balance[v_neighbor] += 0.25 * arc_power
            else:
                Q_balance[node_matrix.bp_idx] += 0.25 * arc_power
        else:
            # No horizontal neighbor: 75% to current, 25% to vertical
            Q_balance[welding_node_idx] += 0.25 * arc_power
            v_neighbor = node_matrix.get_vertical_neighbor(welding_node_idx, -1)
            if v_neighbor is not None:
                Q_balance[v_neighbor] += 0.25 * arc_power
            else:
                Q_balance[node_matrix.bp_idx] += 0.25 * arc_power
    
    # --- Radiation ---
    T_K4 = (T + 273.15)**4
    T_amb_K4 = (AMBIENT_TEMP + 273.15)**4
    top_layer_idx = node_matrix.get_top_layer_idx()
    
    # Table radiation
    if node_matrix.table_idx is not None:
        Q_balance[node_matrix.table_idx] -= (EPSILON_TABLE * STEFAN_BOLTZMANN * 
                                              model.area_table_rad * 
                                              (T_K4[node_matrix.table_idx] - T_amb_K4))
    
    # Base plate radiation (calculate exposed area)
    if node_matrix.bp_idx is not None:
        covered_area = sum(node_matrix.areas[i] for i in range(num_nodes) 
                          if node_matrix.level_type[i] != 'special')
        eff_bp_area = max(0, model.area_bp_total - covered_area)
        Q_balance[node_matrix.bp_idx] -= (EPSILON_BP * STEFAN_BOLTZMANN * 
                                          eff_bp_area * 
                                          (T_K4[node_matrix.bp_idx] - T_amb_K4))
    
    # WAAM nodes radiation (using matrix logic)
    for i in range(num_nodes):
        if node_matrix.level_type[i] == 'special':
            continue
        
        rad_area = node_matrix.compute_radiation_area(i, top_layer_idx)
        epsilon = get_epsilon_waam(T[i])
        Q_balance[i] -= epsilon * STEFAN_BOLTZMANN * rad_area * (T_K4[i] - T_amb_K4)
    
    # --- Conduction ---
    # Table <-> Base plate
    if node_matrix.table_idx is not None and node_matrix.bp_idx is not None:
        q_contact = ALPHA_CONTACT * CONTACT_BP_TABLE * (T[node_matrix.bp_idx] - T[node_matrix.table_idx])
        Q_balance[node_matrix.table_idx] += q_contact
        Q_balance[node_matrix.bp_idx] -= q_contact
    
    # Conduction between all nodes (using neighbor finding)
    for i in range(num_nodes):
        if node_matrix.level_type[i] == 'special' and i != node_matrix.bp_idx:
            continue
        
        # Vertical conduction (upward)
        v_up = node_matrix.get_vertical_neighbor(i, +1)
        if v_up is not None:
            # Determine distance and thermal conductivity
            if i == node_matrix.bp_idx:
                dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
                lam = (LAMBDA_BP + LAMBDA_WAAM) / 2
            else:
                dist = LAYER_HEIGHT
                lam = LAMBDA_WAAM
            
            area = node_matrix.areas[v_up]
            q_vert = lam * area / dist * (T[v_up] - T[i])
            Q_balance[i] += q_vert
            Q_balance[v_up] -= q_vert
        
        # Horizontal conduction (rightward - to avoid double counting)
        h_right = node_matrix.get_horizontal_neighbor(i, +1)
        if h_right is not None:
            overlap_width = TRACK_WIDTH * TRACK_OVERLAP
            contact_area = overlap_width * TRACK_LENGTH
            dist = overlap_width
            q_horiz = LAMBDA_WAAM * contact_area / dist * (T[h_right] - T[i])
            Q_balance[i] += q_horiz
            Q_balance[h_right] -= q_horiz
        
        # Longitudinal conduction (forward - to avoid double counting)
        l_forward = node_matrix.get_longitudinal_neighbor(i, +1)
        if l_forward is not None:
            element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
            bead_width = (TRACK_WIDTH if node_matrix.bead_idx[i] == 0 
                         else TRACK_WIDTH * (1 - TRACK_OVERLAP))
            area = bead_width * LAYER_HEIGHT
            dist = element_length
            q_long = LAMBDA_WAAM * area / dist * (T[l_forward] - T[i])
            Q_balance[i] += q_long
            Q_balance[l_forward] -= q_long
    
    # --- Temperature Update (Euler explicit) ---
    new_T = T.copy()
    
    for i in range(num_nodes):
        if i == node_matrix.table_idx:
            new_T[i] += (Q_balance[i] * DT) / (model.m_table * CP_TABLE)
        elif i == node_matrix.bp_idx:
            cp = get_cp_bp(T[i])
            new_T[i] += (Q_balance[i] * DT) / (model.m_bp * cp)
        elif node_matrix.level_type[i] == 'layer':
            cp = get_cp_waam(T[i])
            new_T[i] += (Q_balance[i] * DT) / (model.m_layer * cp)
        else:  # bead or element level
            cp = get_cp_waam(T[i])
            mass = node_matrix.masses[i]
            new_T[i] += (Q_balance[i] * DT) / (mass * cp)
    
    # Update node_matrix temperatures
    node_matrix.temperatures = new_T.tolist()
    
    return node_matrix


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
    Main simulation function using NodeMatrix for efficient layer/bead/element tracking.
    Uses new matrix-based logic for neighbor finding and radiation calculation.
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
    
    # Check DT stability
    alpha = LAMBDA_WAAM / (RHO_WAAM * CP_WAAM_A)
    dx_min = min(LAYER_HEIGHT, TRACK_WIDTH / max(N_ELEMENTS_PER_BEAD, 1))
    dt_max_stable = dx_min**2 / (2 * alpha)
    if DT > dt_max_stable:
        print(f"Warning: DT ({DT:.3f}s) may be too large for stability. Recommended max: {dt_max_stable:.3f}s")
    
    # Calculate geometry
    effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
    layer_area = effective_layer_width * TRACK_LENGTH
    layer_volume = layer_area * LAYER_HEIGHT
    side_area_layer = 2 * (TRACK_LENGTH * LAYER_HEIGHT + effective_layer_width * LAYER_HEIGHT)
    
    total_weld_distance = NUMBER_OF_TRACKS * TRACK_LENGTH
    layer_duration = total_weld_distance / PROCESS_SPEED
    
    # Bead geometry
    bead_width_effective = TRACK_WIDTH * (1 - TRACK_OVERLAP)
    bead_area_first = TRACK_WIDTH * TRACK_LENGTH
    bead_area_subsequent = bead_width_effective * TRACK_LENGTH
    bead_volume_first = bead_area_first * LAYER_HEIGHT
    bead_volume_subsequent = bead_area_subsequent * LAYER_HEIGHT
    m_bead_first = bead_volume_first * RHO_WAAM
    m_bead_subsequent = bead_volume_subsequent * RHO_WAAM
    overlap_width = TRACK_WIDTH * TRACK_OVERLAP
    bead_contact_area = overlap_width * TRACK_LENGTH
    bead_duration = TRACK_LENGTH / PROCESS_SPEED
    
    # Element geometry (if enabled)
    element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
    element_duration = element_length / PROCESS_SPEED
    
    vol_bp = BP_LENGTH * BP_WIDTH * BP_THICKNESS
    m_bp = vol_bp * RHO_BP
    vol_table = TABLE_LENGTH * TABLE_WIDTH * TABLE_THICKNESS
    m_table = vol_table * RHO_TABLE
    m_layer = layer_volume * RHO_WAAM
    
    bead_params = {
        'bead_area_first': bead_area_first,
        'bead_area_subsequent': bead_area_subsequent,
        'm_bead_first': m_bead_first,
        'm_bead_subsequent': m_bead_subsequent,
        'bead_contact_area': bead_contact_area,
        'bead_duration': bead_duration,
        'overlap_width': overlap_width
    }
    
    thermal_model = ThermalModel(layer_area, side_area_layer, bead_params)
    
    # Initialize NodeMatrix
    node_matrix = NodeMatrix()
    
    # Add table and baseplate nodes
    node_matrix.table_idx = node_matrix.add_node(
        layer_idx=-2, bead_idx=-1, element_idx=-1, level_type='special',
        mass=m_table, area=0.0, temperature=AMBIENT_TEMP
    )
    node_matrix.bp_idx = node_matrix.add_node(
        layer_idx=-1, bead_idx=-1, element_idx=-1, level_type='special',
        mass=m_bp, area=0.0, temperature=AMBIENT_TEMP
    )
    
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
        raise ValueError(f"Arc power ({ARC_POWER:.1f} W) insufficient to melt wire.")
    
    print(f"Starting simulation (Mode {MODE})...")
    print(f"Layer geometry: {effective_layer_width*1000:.1f}mm x {TRACK_LENGTH*1000:.1f}mm x {LAYER_HEIGHT*1000:.1f}mm")
    print(f"Layer duration: {layer_duration:.1f}s ({NUMBER_OF_TRACKS} tracks at {PROCESS_SPEED*1000:.1f}mm/s)")
    print(f"Total height after {NUMBER_OF_LAYERS} layers: {NUMBER_OF_LAYERS * LAYER_HEIGHT*1000:.1f}mm")
    print(f"Discretization: {N_LAYERS_AS_BEADS} top layers as beads, {N_LAYERS_WITH_ELEMENTS} with elements ({N_ELEMENTS_PER_BEAD}/bead)")
    print(f"Arc Power: Total = {ARC_POWER:.1f} W, Wire Melting = {power_wire_melting:.1f} W, Effective = {effective_arc_power:.1f} W")
    
    # Main simulation loop
    for i_layer in tqdm(range(NUMBER_OF_LAYERS)):
        
        # Determine discretization level for this layer
        layers_from_top = NUMBER_OF_LAYERS - i_layer - 1
        use_beads = layers_from_top < N_LAYERS_AS_BEADS
        use_elements = layers_from_top < N_LAYERS_WITH_ELEMENTS
        
        if use_beads:
            # Bead-level or element-level discretization
            for i_bead in range(NUMBER_OF_TRACKS):
                bead_area = bead_area_first if i_bead == 0 else bead_area_subsequent
                bead_mass = m_bead_first if i_bead == 0 else m_bead_subsequent
                
                if use_elements:
                    # Element-level: subdivide bead into elements
                    for i_element in range(N_ELEMENTS_PER_BEAD):
                        element_area = bead_area / N_ELEMENTS_PER_BEAD
                        element_mass = bead_mass / N_ELEMENTS_PER_BEAD
                        
                        welding_node_idx = node_matrix.add_node(
                            layer_idx=i_layer, bead_idx=i_bead, element_idx=i_element,
                            level_type='element', mass=element_mass, area=element_area,
                            temperature=MELTING_TEMP
                        )
                        
                        # Simulate welding of this element
                        steps = int(element_duration / DT)
                        for _ in range(steps):
                            node_matrix = update_temperatures_matrix(
                                node_matrix, thermal_model, is_welding=True,
                                arc_power=effective_arc_power, welding_node_idx=welding_node_idx
                            )
                            current_time += DT
                            
                            logging_counter += 1
                            if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                                log_data(current_time, node_matrix, time_log, temp_layers_log, temp_bp_log, temp_table_log)
                else:
                    # Bead-level: add whole bead
                    welding_node_idx = node_matrix.add_node(
                        layer_idx=i_layer, bead_idx=i_bead, element_idx=-1,
                        level_type='bead', mass=bead_mass, area=bead_area,
                        temperature=MELTING_TEMP
                    )
                    
                    # Simulate welding of this bead
                    steps = int(bead_duration / DT)
                    for _ in range(steps):
                        node_matrix = update_temperatures_matrix(
                            node_matrix, thermal_model, is_welding=True,
                            arc_power=effective_arc_power, welding_node_idx=welding_node_idx
                        )
                        current_time += DT
                        
                        logging_counter += 1
                        if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                            log_data(current_time, node_matrix, time_log, temp_layers_log, temp_bp_log, temp_table_log)
        else:
            # Layer-level: add entire layer as single node
            layer_node_idx = node_matrix.add_node(
                layer_idx=i_layer, bead_idx=-1, element_idx=-1,
                level_type='layer', mass=m_layer, area=layer_area,
                temperature=MELTING_TEMP
            )
            
            # Simulate entire layer deposition
            steps = int(layer_duration / DT)
            for _ in range(steps):
                node_matrix = update_temperatures_matrix(
                    node_matrix, thermal_model, is_welding=True,
                    arc_power=effective_arc_power, welding_node_idx=layer_node_idx
                )
                current_time += DT
                
                logging_counter += 1
                if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                    log_data(current_time, node_matrix, time_log, temp_layers_log, temp_bp_log, temp_table_log)
        
        # Cool until interpass temperature reached
        time_start_wait = current_time
        
        while True:
            # Get max temperature of current layer
            current_layer_nodes = node_matrix.get_nodes_in_layer(i_layer)
            if current_layer_nodes:
                t_hottest = max(node_matrix.temperatures[i] for i in current_layer_nodes)
            else:
                t_hottest = AMBIENT_TEMP
            
            condition_met = t_hottest <= INTERLAYER_TEMP
            
            # Mode 1 special case
            if MODE == 1 and i_layer == 0:
                if current_time - time_start_wait >= MODE_1_WAIT_TIME:
                    break
            elif condition_met:
                break
            
            node_matrix = update_temperatures_matrix(
                node_matrix, thermal_model, is_welding=False
            )
            current_time += DT
            
            logging_counter += 1
            if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                log_data(current_time, node_matrix, time_log, temp_layers_log, temp_bp_log, temp_table_log)
        
        wait_times.append(current_time - time_start_wait)
        
        # Consolidate old layers (beyond N_LAYERS_AS_BEADS) to layer-level
        layers_to_consolidate = i_layer - N_LAYERS_AS_BEADS + 1
        if layers_to_consolidate > 0:
            for layer_to_consolidate in range(layers_to_consolidate):
                nodes_in_layer = node_matrix.get_nodes_in_layer(layer_to_consolidate)
                if len(nodes_in_layer) > 1:
                    # Consolidate multiple nodes into one layer-level node
                    total_mass = sum(node_matrix.masses[i] for i in nodes_in_layer)
                    weighted_temp = sum(node_matrix.masses[i] * node_matrix.temperatures[i] 
                                       for i in nodes_in_layer) / total_mass
                    
                    # Remove old nodes (in reverse to preserve indices)
                    for idx in sorted(nodes_in_layer, reverse=True):
                        del node_matrix.layer_idx[idx]
                        del node_matrix.bead_idx[idx]
                        del node_matrix.element_idx[idx]
                        del node_matrix.level_type[idx]
                        del node_matrix.masses[idx]
                        del node_matrix.areas[idx]
                        del node_matrix.temperatures[idx]
                    
                    # Add consolidated layer node
                    node_matrix.add_node(
                        layer_idx=layer_to_consolidate, bead_idx=-1, element_idx=-1,
                        level_type='layer', mass=m_layer, area=layer_area,
                        temperature=weighted_temp
                    )
    
    return time_log, temp_layers_log, temp_bp_log, temp_table_log, wait_times

def log_data(t, node_matrix, t_log, layers_log, bp_log, temp_table_log):
    """Log temperature data from NodeMatrix."""
    t_log.append(t)
    
    # Extract layer temperatures (group by layer and take max)
    top_layer_idx = node_matrix.get_top_layer_idx()
    layer_temps = []
    for layer_idx in range(top_layer_idx + 1):
        nodes_in_layer = node_matrix.get_nodes_in_layer(layer_idx)
        if nodes_in_layer:
            max_temp = max(node_matrix.temperatures[i] for i in nodes_in_layer)
            layer_temps.append(max_temp)
    
    layers_log.append(layer_temps)
    bp_log.append(node_matrix.temperatures[node_matrix.bp_idx] if node_matrix.bp_idx is not None else AMBIENT_TEMP)
    temp_table_log.append(node_matrix.temperatures[node_matrix.table_idx] if node_matrix.table_idx is not None else AMBIENT_TEMP)

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