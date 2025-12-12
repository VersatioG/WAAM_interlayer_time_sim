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
N_LAYERS_AS_BEADS = 4       # Number of top layers modeled as individual beads (default: 2)
N_LAYERS_WITH_ELEMENTS = 2  # Number of top layers where beads are subdivided into elements (0 = disabled)
N_ELEMENTS_PER_BEAD = 5     # Number of elements per bead along track length (if enabled)

# --- WAAM Process Parameters ---
NUMBER_OF_LAYERS = 20
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
ARC_POWER_CURRENT_FRACTION = 0.5  # Fraction of arc power going to current node (0.0-1.0)
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
    
    def get_nodes_in_bead(self, layer_idx, bead_idx):
        """Get all element nodes within a specific bead."""
        return [i for i in range(len(self.layer_idx))
                if self.layer_idx[i] == layer_idx and 
                   self.bead_idx[i] == bead_idx and
                   self.level_type[i] == 'element']
    
    def get_layer_level_type(self, layer_idx):
        """Get the discretization level type for a specific layer."""
        for i in range(len(self.layer_idx)):
            if self.layer_idx[i] == layer_idx and self.level_type[i] != 'special':
                return self.level_type[i]
        return None
    
    def get_vertical_neighbor(self, node_idx, direction):
        """
        Get vertical neighbor (±1 in layer direction).
        direction: +1 for above, -1 for below
        Returns node index or None if not found.
        
        For same-level neighbors, returns direct match.
        For cross-level neighbors, returns None (use get_cross_level_vertical_info instead).
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        level = self.level_type[node_idx]
        
        target_layer = layer + direction
        
        # Special case: going below layer 0 leads to base plate
        if target_layer < 0:
            return self.bp_idx
        
        # Find matching node in target layer (same level)
        for i in range(len(self.layer_idx)):
            if (self.layer_idx[i] == target_layer and 
                self.bead_idx[i] == bead and 
                self.element_idx[i] == element and
                self.level_type[i] == level):
                return i
        
        # If layer-level node, check for bead-level layer below/above
        if level == 'layer':
            # No direct match found, cross-level will be handled separately
            return None
        
        return None
    
    def get_cross_level_vertical_info(self, node_idx, direction):
        """
        Get information for cross-level vertical heat transfer.
        
        Returns dict with:
        - 'type': 'same', 'element_to_bead', 'bead_to_element', 'bead_to_layer', 'layer_to_bead', or 'none'
        - 'target_nodes': list of node indices in target layer
        - 'source_nodes': list of node indices that should be aggregated (for mass-weighted avg)
        - 'target_layer': layer index of target
        - 'target_level': level type of target layer
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        level = self.level_type[node_idx]
        
        target_layer = layer + direction
        
        # Special case: base plate
        if target_layer < 0:
            return {'type': 'to_baseplate', 'target_nodes': [self.bp_idx], 
                    'source_nodes': [node_idx], 'target_layer': -1, 'target_level': 'special'}
        
        # Get target layer's level type
        target_level = self.get_layer_level_type(target_layer)
        
        if target_level is None:
            return {'type': 'none', 'target_nodes': [], 'source_nodes': [], 
                    'target_layer': target_layer, 'target_level': None}
        
        # Same level - direct transfer
        if level == target_level:
            neighbor = self.get_vertical_neighbor(node_idx, direction)
            if neighbor is not None:
                return {'type': 'same', 'target_nodes': [neighbor], 
                        'source_nodes': [node_idx], 'target_layer': target_layer, 
                        'target_level': target_level}
            return {'type': 'none', 'target_nodes': [], 'source_nodes': [], 
                    'target_layer': target_layer, 'target_level': target_level}
        
        # Element → Bead (finer to coarser)
        if level == 'element' and target_level == 'bead':
            # Find the corresponding bead in target layer
            target_bead_nodes = [i for i in range(len(self.layer_idx))
                                 if self.layer_idx[i] == target_layer and 
                                    self.bead_idx[i] == bead and
                                    self.level_type[i] == 'bead']
            # Source: all elements in same bead (for mass-weighted aggregation)
            source_elements = self.get_nodes_in_bead(layer, bead)
            return {'type': 'element_to_bead', 'target_nodes': target_bead_nodes,
                    'source_nodes': source_elements, 'target_layer': target_layer,
                    'target_level': target_level}
        
        # Bead → Element (coarser to finer)
        if level == 'bead' and target_level == 'element':
            # Find all elements in the corresponding bead of target layer
            target_elements = self.get_nodes_in_bead(target_layer, bead)
            return {'type': 'bead_to_element', 'target_nodes': target_elements,
                    'source_nodes': [node_idx], 'target_layer': target_layer,
                    'target_level': target_level}
        
        # Bead → Layer (finer to coarser)
        if level == 'bead' and target_level == 'layer':
            # Find layer-level node in target layer
            target_layer_nodes = [i for i in range(len(self.layer_idx))
                                  if self.layer_idx[i] == target_layer and 
                                     self.level_type[i] == 'layer']
            # Source: all beads in same layer (for mass-weighted aggregation)
            source_beads = [i for i in range(len(self.layer_idx))
                           if self.layer_idx[i] == layer and self.level_type[i] == 'bead']
            return {'type': 'bead_to_layer', 'target_nodes': target_layer_nodes,
                    'source_nodes': source_beads, 'target_layer': target_layer,
                    'target_level': target_level}
        
        # Layer → Bead (coarser to finer)
        if level == 'layer' and target_level == 'bead':
            # Find all beads in target layer
            target_beads = [i for i in range(len(self.layer_idx))
                           if self.layer_idx[i] == target_layer and self.level_type[i] == 'bead']
            return {'type': 'layer_to_bead', 'target_nodes': target_beads,
                    'source_nodes': [node_idx], 'target_layer': target_layer,
                    'target_level': target_level}
        
        # Element → Layer (skip bead level)
        if level == 'element' and target_level == 'layer':
            target_layer_nodes = [i for i in range(len(self.layer_idx))
                                  if self.layer_idx[i] == target_layer and 
                                     self.level_type[i] == 'layer']
            # All elements in the layer
            source_elements = [i for i in range(len(self.layer_idx))
                              if self.layer_idx[i] == layer and self.level_type[i] == 'element']
            return {'type': 'element_to_layer', 'target_nodes': target_layer_nodes,
                    'source_nodes': source_elements, 'target_layer': target_layer,
                    'target_level': target_level}
        
        # Layer → Element (skip bead level)
        if level == 'layer' and target_level == 'element':
            target_elements = [i for i in range(len(self.layer_idx))
                              if self.layer_idx[i] == target_layer and self.level_type[i] == 'element']
            return {'type': 'layer_to_element', 'target_nodes': target_elements,
                    'source_nodes': [node_idx], 'target_layer': target_layer,
                    'target_level': target_level}
        
        return {'type': 'none', 'target_nodes': [], 'source_nodes': [], 
                'target_layer': target_layer, 'target_level': target_level}
    
    def compute_mass_weighted_temperature(self, node_indices, temperatures):
        """Compute mass-weighted average temperature for a list of nodes."""
        if not node_indices:
            return 0.0
        total_mass = sum(self.masses[i] for i in node_indices)
        if total_mass == 0:
            return temperatures[node_indices[0]] if node_indices else 0.0
        weighted_sum = sum(self.masses[i] * temperatures[i] for i in node_indices)
        return weighted_sum / total_mass
    
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
    
    def get_beads_in_layer(self, layer_idx):
        """Get list of unique bead indices in a layer."""
        beads = set()
        for i in range(len(self.layer_idx)):
            if self.layer_idx[i] == layer_idx and self.bead_idx[i] >= 0:
                beads.add(self.bead_idx[i])
        return sorted(list(beads))
    
    def get_elements_in_bead(self, layer_idx, bead_idx):
        """Get list of element indices in a specific bead."""
        elements = set()
        for i in range(len(self.layer_idx)):
            if (self.layer_idx[i] == layer_idx and 
                self.bead_idx[i] == bead_idx and 
                self.element_idx[i] >= 0):
                elements.add(self.element_idx[i])
        return sorted(list(elements))
    
    def get_elements_placed_in_bead(self, layer_idx, bead_idx):
        """Get the number of elements already placed in a specific bead position."""
        return len(self.get_elements_in_bead(layer_idx, bead_idx))
    
    def get_max_bead_in_layer(self, layer_idx):
        """Get the maximum bead index currently existing in a layer."""
        max_bead = -1
        for i in range(len(self.layer_idx)):
            if self.layer_idx[i] == layer_idx and self.bead_idx[i] > max_bead:
                max_bead = self.bead_idx[i]
        return max_bead
    
    def get_max_element_in_bead(self, layer_idx, bead_idx):
        """Get the maximum element index currently existing in a bead."""
        max_elem = -1
        for i in range(len(self.layer_idx)):
            if (self.layer_idx[i] == layer_idx and 
                self.bead_idx[i] == bead_idx and 
                self.element_idx[i] > max_elem):
                max_elem = self.element_idx[i]
        return max_elem
    
    def compute_radiation_area(self, node_idx, top_layer_idx):
        """
        Compute radiation area for a node based on its position in the matrix.
        
        Logic:
        - Layer-level: Side surfaces always, top surface only if top layer
        - Bead-level: First/last bead radiates long side + both short sides,
                      inner beads radiate only short sides.
                      In top layer: check for missing neighbors and radiate exposed faces.
        - Element-level: First/last element in bead radiates broad side,
                        first/last bead radiates long side (proportional).
                        In top layer: check for missing neighbors.
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        level = self.level_type[node_idx]
        
        if level == 'special':
            return 0.0
        
        rad_area = 0.0
        is_top = (layer == top_layer_idx)
        
        # Calculate geometry parameters
        effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
        
        if level == 'layer':
            # Layer-level: always side surfaces, top only if top layer
            layer_area = effective_layer_width * TRACK_LENGTH
            side_area = 2 * (TRACK_LENGTH + effective_layer_width) * LAYER_HEIGHT
            
            if is_top:
                rad_area += layer_area  # Top surface
            rad_area += side_area  # All side surfaces
            
        elif level == 'bead':
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            bead_top_area = bead_width * TRACK_LENGTH
            
            # Get current bead count and max bead in this layer
            beads_in_layer = self.get_beads_in_layer(layer)
            max_bead = max(beads_in_layer) if beads_in_layer else 0
            min_bead = min(beads_in_layer) if beads_in_layer else 0
            
            # Top surface: only if top layer
            if is_top:
                rad_area += bead_top_area
            
            # Side surfaces:
            # Long sides (parallel to track, perpendicular to bead direction)
            if bead == min_bead:
                # First bead: radiates left long side
                rad_area += TRACK_LENGTH * LAYER_HEIGHT
            if bead == max_bead:
                # Last bead: radiates right long side
                rad_area += TRACK_LENGTH * LAYER_HEIGHT
            
            # Short sides (perpendicular to track, at start/end of track)
            # Both short sides always radiate for all beads
            rad_area += 2 * bead_width * LAYER_HEIGHT
            
            # In top layer: check for missing horizontal neighbors
            if is_top:
                # Check left neighbor (bead - 1)
                if bead > 0:
                    h_left = self.get_horizontal_neighbor(node_idx, -1)
                    if h_left is None:
                        # No left neighbor exists yet -> radiate that face
                        rad_area += TRACK_LENGTH * LAYER_HEIGHT
                # Check right neighbor (bead + 1)
                h_right = self.get_horizontal_neighbor(node_idx, +1)
                if h_right is None and bead < NUMBER_OF_TRACKS - 1:
                    # Right neighbor doesn't exist yet -> radiate that face
                    rad_area += TRACK_LENGTH * LAYER_HEIGHT
        
        elif level == 'element':
            element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            element_top_area = element_length * bead_width
            
            # Get current element/bead counts
            beads_in_layer = self.get_beads_in_layer(layer)
            max_bead = max(beads_in_layer) if beads_in_layer else 0
            min_bead = min(beads_in_layer) if beads_in_layer else 0
            
            max_element = self.get_max_element_in_bead(layer, bead)
            
            # Top surface: only if top layer
            if is_top:
                rad_area += element_top_area
            
            # Broad sides (perpendicular to track direction, at element start/end)
            # First element in bead: radiates start face
            if element == 0:
                rad_area += bead_width * LAYER_HEIGHT
            # Last element in bead: radiates end face
            if element == max_element:
                rad_area += bead_width * LAYER_HEIGHT
            
            # Long sides (parallel to track direction)
            # First bead: elements radiate left long side
            if bead == min_bead:
                rad_area += element_length * LAYER_HEIGHT
            # Last bead: elements radiate right long side
            if bead == max_bead:
                rad_area += element_length * LAYER_HEIGHT
            
            # In top layer: check for missing neighbors
            if is_top:
                # Check horizontal neighbor (left/right bead)
                if bead > 0:
                    h_left = self.get_horizontal_neighbor(node_idx, -1)
                    if h_left is None:
                        rad_area += element_length * LAYER_HEIGHT
                if bead < NUMBER_OF_TRACKS - 1:
                    h_right = self.get_horizontal_neighbor(node_idx, +1)
                    if h_right is None:
                        rad_area += element_length * LAYER_HEIGHT
                
                # Check longitudinal neighbor (along track)
                if element > 0:
                    l_back = self.get_longitudinal_neighbor(node_idx, -1)
                    if l_back is None:
                        rad_area += bead_width * LAYER_HEIGHT
                if element < N_ELEMENTS_PER_BEAD - 1:
                    l_forward = self.get_longitudinal_neighbor(node_idx, +1)
                    if l_forward is None:
                        rad_area += bead_width * LAYER_HEIGHT
        
        return rad_area
    
    def compute_bp_radiation_area(self, top_layer_idx):
        """
        Compute base plate radiation area.
        
        - If no layers yet: full BP surface
        - Layer 0 (first layer): BP top surface minus currently covered area by beads/elements
        - Layer >= 1: BP bottom + 4 sides (top is fully covered)
        """
        if top_layer_idx < 0:
            # No layers yet: full BP surface
            return (2 * BP_LENGTH * BP_WIDTH + 
                    2 * BP_LENGTH * BP_THICKNESS + 
                    2 * BP_WIDTH * BP_THICKNESS)
        
        # Calculate covered area on BP (by first layer nodes)
        first_layer_nodes = self.get_nodes_in_layer(0)
        covered_area = sum(self.areas[i] for i in first_layer_nodes)
        
        if top_layer_idx == 0:
            # Only first layer exists: BP radiates uncovered top + sides
            bp_top_uncovered = max(0, BP_LENGTH * BP_WIDTH - covered_area)
            bp_sides = 2 * BP_LENGTH * BP_THICKNESS + 2 * BP_WIDTH * BP_THICKNESS
            # Bottom is on table, doesn't radiate
            return bp_top_uncovered + bp_sides
        else:
            # Layer >= 1: BP top is fully covered, radiate sides only
            # (Bottom is on table)
            bp_sides = 2 * BP_LENGTH * BP_THICKNESS + 2 * BP_WIDTH * BP_THICKNESS
            return bp_sides
    
    def compute_table_radiation_area(self):
        """
        Compute table radiation area.
        Table radiates with full surface minus BP contact area.
        """
        table_total = (2 * TABLE_LENGTH * TABLE_WIDTH + 
                       2 * TABLE_LENGTH * TABLE_THICKNESS + 
                       2 * TABLE_WIDTH * TABLE_THICKNESS)
        # Subtract BP bottom (contact area)
        return table_total - CONTACT_BP_TABLE
    
    def compute_previous_layer_uncovered_top(self, layer_idx, current_bead_idx, current_element_idx):
        """
        Compute the uncovered top area of the layer below that should still radiate.
        
        This is needed when building on a coarser discretization level.
        Returns the area of the previous layer's top that is not yet covered
        by the current layer's beads/elements.
        """
        if layer_idx <= 0:
            return 0.0
        
        prev_layer = layer_idx - 1
        prev_level = self.get_layer_level_type(prev_layer)
        curr_level = self.get_layer_level_type(layer_idx)
        
        if prev_level is None:
            return 0.0
        
        # Calculate what's covered by current layer's existing nodes
        current_layer_nodes = self.get_nodes_in_layer(layer_idx)
        current_covered = sum(self.areas[i] for i in current_layer_nodes)
        
        # Calculate total top area of previous layer
        prev_layer_nodes = self.get_nodes_in_layer(prev_layer)
        prev_total_top = sum(self.areas[i] for i in prev_layer_nodes)
        
        # Uncovered area
        return max(0.0, prev_total_top - current_covered)
        
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
        # ARC_POWER_CURRENT_FRACTION to current node
        current_power = ARC_POWER_CURRENT_FRACTION * arc_power
        Q_balance[welding_node_idx] += current_power
        
        # Collect neighbors for the remaining power
        neighbors = []
        contact_areas = []
        
        remaining_power = (1.0 - ARC_POWER_CURRENT_FRACTION) * arc_power
        
        # Horizontal neighbor (previous bead in same layer)
        h_neighbor = node_matrix.get_horizontal_neighbor(welding_node_idx, -1)
        if h_neighbor is not None:
            neighbors.append(h_neighbor)
            # Contact area for horizontal: overlap_width * TRACK_LENGTH
            contact_areas.append(TRACK_WIDTH * TRACK_OVERLAP * TRACK_LENGTH)
        
        # Vertical neighbor (layer below)
        v_neighbor = node_matrix.get_vertical_neighbor(welding_node_idx, -1)
        if v_neighbor is not None:
            neighbors.append(v_neighbor)
            # Contact area for vertical: top area of the vertical neighbor
            contact_areas.append(node_matrix.areas[v_neighbor])
        elif node_matrix.bp_idx is not None:
            # If no vertical neighbor, use base plate
            neighbors.append(node_matrix.bp_idx)
            # Contact area for BP: top area of current node
            contact_areas.append(node_matrix.areas[welding_node_idx])
        
        # Special case: if this is the first node in the layer (bead_idx == 0 and no horizontal neighbor)
        is_first_node = (node_matrix.bead_idx[welding_node_idx] == 0 and 
                        node_matrix.get_horizontal_neighbor(welding_node_idx, -1) is None)
        
        if is_first_node and node_matrix.bp_idx in neighbors:
            # First node: entire remaining power goes to BP
            bp_idx = neighbors.index(node_matrix.bp_idx)
            Q_balance[node_matrix.bp_idx] += remaining_power
        elif neighbors:
            # Distribute remaining power among neighbors proportional to contact areas
            total_contact_area = sum(contact_areas)
            if total_contact_area > 0:
                for i, neighbor_idx in enumerate(neighbors):
                    fraction = contact_areas[i] / total_contact_area
                    Q_balance[neighbor_idx] += remaining_power * fraction
            else:
                # If all contact areas are 0, distribute equally
                equal_share = remaining_power / len(neighbors)
                for neighbor_idx in neighbors:
                    Q_balance[neighbor_idx] += equal_share
    
    # --- Radiation ---
    T_K4 = (T + 273.15)**4
    T_amb_K4 = (AMBIENT_TEMP + 273.15)**4
    top_layer_idx = node_matrix.get_top_layer_idx()
    
    # Table radiation (full surface minus BP contact area)
    if node_matrix.table_idx is not None:
        table_rad_area = node_matrix.compute_table_radiation_area()
        Q_balance[node_matrix.table_idx] -= (EPSILON_TABLE * STEFAN_BOLTZMANN * 
                                              table_rad_area * 
                                              (T_K4[node_matrix.table_idx] - T_amb_K4))
    
    # Base plate radiation (using new method that considers layer coverage)
    if node_matrix.bp_idx is not None:
        bp_rad_area = node_matrix.compute_bp_radiation_area(top_layer_idx)
        Q_balance[node_matrix.bp_idx] -= (EPSILON_BP * STEFAN_BOLTZMANN * 
                                          bp_rad_area * 
                                          (T_K4[node_matrix.bp_idx] - T_amb_K4))
    
    # Previous layer uncovered top radiation (when current layer partially covers it)
    # This handles the case where beads/elements are being deposited and 
    # the previous layer's uncovered portions still radiate
    if top_layer_idx > 0:
        prev_layer_idx = top_layer_idx - 1
        prev_level = node_matrix.get_layer_level_type(prev_layer_idx)
        
        if prev_level == 'bead' or prev_level == 'element':
            # For bead/element level in previous layer, check each node individually
            prev_layer_nodes = node_matrix.get_nodes_in_layer(prev_layer_idx)
            
            for i in prev_layer_nodes:
                # Check if this node has a vertical neighbor above
                v_neighbor = node_matrix.get_vertical_neighbor(i, +1)
                
                if v_neighbor is None:
                    # No vertical neighbor above, so this node radiates its top surface
                    rad_area = 0.0
                    
                    if node_matrix.level_type[i] == 'bead':
                        # Bead top area
                        bead_width = TRACK_WIDTH if node_matrix.bead_idx[i] == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
                        bead_top_area = bead_width * TRACK_LENGTH
                        
                        # Special case: if N_LAYERS_WITH_ELEMENTS == 1 and previous layer is bead level
                        # Subtract the top areas of already placed elements on this bead
                        if N_LAYERS_WITH_ELEMENTS == 1 and prev_level == 'bead':
                            elements_placed = node_matrix.get_elements_placed_in_bead(prev_layer_idx, node_matrix.bead_idx[i])
                            element_top_area = (TRACK_LENGTH / N_ELEMENTS_PER_BEAD) * bead_width
                            covered_by_elements = elements_placed * element_top_area
                            rad_area = max(0.0, bead_top_area - covered_by_elements)
                        else:
                            rad_area = bead_top_area
                    
                    elif node_matrix.level_type[i] == 'element':
                        # Element top area
                        element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
                        bead_width = TRACK_WIDTH if node_matrix.bead_idx[i] == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
                        rad_area = element_length * bead_width
                    
                    elif node_matrix.level_type[i] == 'layer':
                        # Layer top area
                        effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
                        rad_area = effective_layer_width * TRACK_LENGTH
                    
                    if rad_area > 0:
                        epsilon = get_epsilon_waam(T[i])
                        Q_balance[i] -= epsilon * STEFAN_BOLTZMANN * rad_area * (T_K4[i] - T_amb_K4)
        
        elif prev_level == 'layer':
            # For layer level, use the old logic (full uncovered area)
            prev_layer_nodes = node_matrix.get_nodes_in_layer(prev_layer_idx)
            if prev_layer_nodes:
                current_layer_nodes = node_matrix.get_nodes_in_layer(top_layer_idx)
                current_covered = sum(node_matrix.areas[i] for i in current_layer_nodes)
                prev_total_top = sum(node_matrix.areas[i] for i in prev_layer_nodes)
                uncovered_top = max(0.0, prev_total_top - current_covered)
                
                if uncovered_top > 0:
                    # Distribute uncovered radiation among previous layer nodes proportionally
                    for i in prev_layer_nodes:
                        node_fraction = node_matrix.areas[i] / prev_total_top if prev_total_top > 0 else 0
                        node_uncovered = uncovered_top * node_fraction
                        epsilon = get_epsilon_waam(T[i])
                        Q_balance[i] -= epsilon * STEFAN_BOLTZMANN * node_uncovered * (T_K4[i] - T_amb_K4)
    
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
    
    # Conduction between all nodes (using neighbor finding with cross-level support)
    processed_pairs = set()  # Track processed node pairs to avoid double counting
    
    # Special case: Base plate to first layer (possibly multi-level)
    if node_matrix.bp_idx is not None:
        bp_idx = node_matrix.bp_idx
        first_layer_nodes = node_matrix.get_nodes_in_layer(0)
        
        if first_layer_nodes:
            # Distance from BP center to first layer center
            dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
            lam = (LAMBDA_BP + LAMBDA_WAAM) / 2
            
            for target_idx in first_layer_nodes:
                pair_key = (min(bp_idx, target_idx), max(bp_idx, target_idx))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                area = node_matrix.areas[target_idx]  # Use WAAM node's area
                q_vert = lam * area / dist * (T[target_idx] - T[bp_idx])
                Q_balance[bp_idx] += q_vert
                Q_balance[target_idx] -= q_vert
    
    for i in range(num_nodes):
        if node_matrix.level_type[i] == 'special':
            continue  # BP already handled above
        
        # Vertical conduction (upward) with cross-level support
        cross_info = node_matrix.get_cross_level_vertical_info(i, +1)
        transfer_type = cross_info['type']
        
        if transfer_type == 'to_baseplate':
            # Already handled in BP section above
            continue
        elif transfer_type == 'none' or transfer_type is None:
            # No vertical neighbor (top layer)
            continue
        elif transfer_type == 'same':
            # Same discretization level - direct transfer
            v_up = cross_info['target_nodes'][0]
            pair_key = (min(i, v_up), max(i, v_up))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            dist = LAYER_HEIGHT
            lam = LAMBDA_WAAM
            
            area = node_matrix.areas[v_up]
            q_vert = lam * area / dist * (T[v_up] - T[i])
            Q_balance[i] += q_vert
            Q_balance[v_up] -= q_vert
            
        elif transfer_type in ['element_to_bead', 'bead_to_layer', 'element_to_layer']:
            # Finer -> Coarser: Multiple source nodes contribute to one target
            target_nodes = cross_info['target_nodes']
            
            if not target_nodes:
                continue
                
            target_idx = target_nodes[0]
            
            # For finer to coarser, compute heat from this source only
            dist = LAYER_HEIGHT
            lam = LAMBDA_WAAM
            area = node_matrix.areas[i]  # Use source node's area
            q_vert = lam * area / dist * (T[target_idx] - T[i])
            Q_balance[i] += q_vert
            Q_balance[target_idx] -= q_vert
            
        elif transfer_type in ['bead_to_element', 'layer_to_bead', 'layer_to_element']:
            # Coarser -> Finer: One source distributes to multiple targets
            target_nodes = cross_info['target_nodes']
            
            if not target_nodes:
                continue
            
            # Process each target (finer node)
            for target_idx in target_nodes:
                pair_key = (min(i, target_idx), max(i, target_idx))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                dist = LAYER_HEIGHT
                lam = LAMBDA_WAAM
                area = node_matrix.areas[target_idx]  # Use target (finer) node's area
                q_vert = lam * area / dist * (T[target_idx] - T[i])
                Q_balance[i] += q_vert
                Q_balance[target_idx] -= q_vert
        
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
    
    if not (0.0 <= ARC_POWER_CURRENT_FRACTION <= 1.0):
        raise ValueError(f"ARC_POWER_CURRENT_FRACTION ({ARC_POWER_CURRENT_FRACTION}) must be between 0.0 and 1.0")
    
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
        
        # Dynamic discretization: based on current number of layers (i_layer + 1)
        # The current layer being deposited is always the "top" layer
        # use_beads: True if current layer should use bead-level (always for top N_LAYERS_AS_BEADS)
        # use_elements: True if current layer should use element-level (always for top N_LAYERS_WITH_ELEMENTS)
        current_num_layers = i_layer + 1  # Total layers including this one
        
        # Current layer is always the topmost, so it always uses the finest discretization
        use_beads = N_LAYERS_AS_BEADS >= 1  # Current layer uses beads if N_LAYERS_AS_BEADS >= 1
        use_elements = N_LAYERS_WITH_ELEMENTS >= 1  # Current layer uses elements if N_LAYERS_WITH_ELEMENTS >= 1
        
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
        
        # Dynamic consolidation based on current layer count
        # Elements → Beads: for layers beyond N_LAYERS_WITH_ELEMENTS from top
        # Beads → Layer: for layers beyond N_LAYERS_AS_BEADS from top
        
        current_num_layers = i_layer + 1
        
        # Consolidate elements to beads for layers that should now be bead-level only
        if N_LAYERS_WITH_ELEMENTS > 0:
            for layer_idx in range(current_num_layers):
                layers_from_current_top = current_num_layers - layer_idx - 1
                should_be_elements = layers_from_current_top < N_LAYERS_WITH_ELEMENTS
                
                if not should_be_elements:
                    # This layer should NOT have elements anymore, consolidate to beads
                    current_level = node_matrix.get_layer_level_type(layer_idx)
                    if current_level == 'element':
                        # Get all beads that have elements
                        beads_in_layer = node_matrix.get_beads_in_layer(layer_idx)
                        for bead_idx in beads_in_layer:
                            element_nodes = node_matrix.get_nodes_in_bead(layer_idx, bead_idx)
                            if len(element_nodes) > 0:
                                # Consolidate elements into single bead
                                total_mass = sum(node_matrix.masses[i] for i in element_nodes)
                                weighted_temp = sum(node_matrix.masses[i] * node_matrix.temperatures[i] 
                                                   for i in element_nodes) / total_mass if total_mass > 0 else AMBIENT_TEMP
                                
                                bead_area = bead_area_first if bead_idx == 0 else bead_area_subsequent
                                bead_mass = m_bead_first if bead_idx == 0 else m_bead_subsequent
                                
                                # Remove element nodes (in reverse to preserve indices)
                                for idx in sorted(element_nodes, reverse=True):
                                    del node_matrix.layer_idx[idx]
                                    del node_matrix.bead_idx[idx]
                                    del node_matrix.element_idx[idx]
                                    del node_matrix.level_type[idx]
                                    del node_matrix.masses[idx]
                                    del node_matrix.areas[idx]
                                    del node_matrix.temperatures[idx]
                                
                                # Update special indices after deletion
                                if node_matrix.table_idx is not None:
                                    node_matrix.table_idx = next((i for i, lt in enumerate(node_matrix.level_type) 
                                                                  if lt == 'special' and node_matrix.layer_idx[i] == -2), None)
                                if node_matrix.bp_idx is not None:
                                    node_matrix.bp_idx = next((i for i, lt in enumerate(node_matrix.level_type) 
                                                               if lt == 'special' and node_matrix.layer_idx[i] == -1), None)
                                
                                # Add consolidated bead node
                                node_matrix.add_node(
                                    layer_idx=layer_idx, bead_idx=bead_idx, element_idx=-1,
                                    level_type='bead', mass=bead_mass, area=bead_area,
                                    temperature=weighted_temp
                                )
        
        # Consolidate beads to layer for layers that should now be layer-level
        for layer_idx in range(current_num_layers):
            layers_from_current_top = current_num_layers - layer_idx - 1
            should_be_beads = layers_from_current_top < N_LAYERS_AS_BEADS
            
            if not should_be_beads:
                # This layer should NOT have beads anymore, consolidate to layer
                nodes_in_layer = node_matrix.get_nodes_in_layer(layer_idx)
                if len(nodes_in_layer) > 1:
                    # Consolidate multiple nodes into one layer-level node
                    total_mass = sum(node_matrix.masses[i] for i in nodes_in_layer)
                    weighted_temp = sum(node_matrix.masses[i] * node_matrix.temperatures[i] 
                                       for i in nodes_in_layer) / total_mass if total_mass > 0 else AMBIENT_TEMP
                    
                    # Remove old nodes (in reverse to preserve indices)
                    for idx in sorted(nodes_in_layer, reverse=True):
                        del node_matrix.layer_idx[idx]
                        del node_matrix.bead_idx[idx]
                        del node_matrix.element_idx[idx]
                        del node_matrix.level_type[idx]
                        del node_matrix.masses[idx]
                        del node_matrix.areas[idx]
                        del node_matrix.temperatures[idx]
                    
                    # Update special indices after deletion
                    if node_matrix.table_idx is not None:
                        node_matrix.table_idx = next((i for i, lt in enumerate(node_matrix.level_type) 
                                                      if lt == 'special' and node_matrix.layer_idx[i] == -2), None)
                    if node_matrix.bp_idx is not None:
                        node_matrix.bp_idx = next((i for i, lt in enumerate(node_matrix.level_type) 
                                                   if lt == 'special' and node_matrix.layer_idx[i] == -1), None)
                    
                    # Add consolidated layer node
                    node_matrix.add_node(
                        layer_idx=layer_idx, bead_idx=-1, element_idx=-1,
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