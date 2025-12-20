import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from Material_Properties import get_material

# =============================================================================
# INPUT BLOCK (Adjust values here)
# =============================================================================

# --- Simulation Settings ---
DT = 0.02   # Simulation time step [s] (Smaller = more accurate, but slower)
LOGGING_EVERY_N_STEPS = 10  # Log data every N time steps to reduce memory usage and plotting overhead

# --- Discretization Settings (counted from top) ---
# NOTE: Element-level discretization increases computational cost significantly.
# N_LAYERS_AS_BEADS: Top layers where each track is modeled as individual bead (thermal node)
# N_LAYERS_WITH_ELEMENTS: Top layers where beads are further subdivided (finer resolution)
# N_ELEMENTS_PER_BEAD: Subdivision count per bead (only used if N_LAYERS_WITH_ELEMENTS > 0)
# Currently the simulation uses N_LAYERS_AS_BEADS = 2 (current + previous layer as beads)
# Element-level refinement is prepared but not yet fully implemented
N_LAYERS_AS_BEADS = 2       # Number of top layers modeled as individual beads (default: 2)
N_LAYERS_WITH_ELEMENTS = 0  # Number of top layers where beads are subdivided into elements (0 = disabled)
N_ELEMENTS_PER_BEAD = 5     # Number of elements per bead along track length (if enabled)

# --- WAAM Process Parameters ---
NUMBER_OF_LAYERS = 15
LAYER_HEIGHT = 0.0023       # [m] (e.g., 2.4mm)

# Track geometry
TRACK_WIDTH = 0.0053         # [m] Width of a single weld track (bead width)
TRACK_OVERLAP = 0.738        # Center distance ratio (Programmed distance / TRACK_WIDTH)
NUMBER_OF_TRACKS = 5        # Number of parallel tracks per layer
TRACK_LENGTH = 0.285         # [m] Length of each track

# Process parameters
PROCESS_SPEED = 0.010        # [m/s] Welding speed (travel speed)
MELTING_TEMP = 1450.0       # [°C] Temperature at which the wire impacts
INTERLAYER_TEMP = 200.0     # [°C] Max. temp of previous layer before starting next
ARC_POWER = 4760.0          # [W] Total arc power during welding
ARC_POWER_CURRENT_FRACTION = 0.4  # Fraction of arc power going to current node (0.0-1.0)
WIRE_FEED_RATE = 0.08       # [m/s] Wire feed rate (typical: 0.03-0.08 m/s)
WIRE_DIAMETER = 0.0012      # [m] Wire diameter (e.g., 1.2mm)

# --- Robot Logic ---
# Robot_Wait = Base_Time + (Layer_Index * Factor)  # Linear
# Or cubic: Robot_Wait = a + b*i + c*i^2 + d*i^3
ROBOT_FIT_MODE = "cubic"  # "linear" or "cubic"

# --- Geometry & Material: WAAM Wire ---
MATERIAL_WAAM_NAME = "S235JR"
RHO_WAAM = 7800.0          # [kg/m^3] Density
LAMBDA_WAAM = 45.0         # [W/(m K)] Thermal conductivity
EPSILON_WAAM = 0.6         # Emissivity (solid)
EPSILON_WAAM_LIQUID = 0.3  # Emissivity (liquid/molten)

# --- Geometry & Material: Base Plate ---
BP_LENGTH = 0.15           # [m]
BP_WIDTH = 0.15            # [m]
BP_THICKNESS = 0.01        # [m]

MATERIAL_BP_NAME = "S235JR"
RHO_BP = 7850.0             # [kg/m^3]
LAMBDA_BP = 45.0            # [W/(m K)]
EPSILON_BP = 0.8            # Emissivity

# --- Geometry & Material: Welding Table ---
TABLE_LENGTH = 2           # [m]
TABLE_WIDTH = 1.2          # [m]
TABLE_THICKNESS = 0.01     # [m]

MATERIAL_TABLE_NAME = "S235JR"
RHO_TABLE = 7850.0         # [kg/m^3]
LAMBDA_TABLE = 45.0        # [W/(m K)]
EPSILON_TABLE = 0.8        # Emissivity

# --- Table Discretization Settings ---
# TABLE_DISCRETIZATION_MODE controls how finely the table is subdivided:
#   Mode 0: Table as single node (fastest, least accurate)
#   Mode 1: Table subdivided into N_TABLE_X * N_TABLE_Y * N_TABLE_Z nodes (base subdivision)
#   Mode 2+: Each higher mode adds +1 to each dimension subdivision
# The base plate is placed on one of the top corner nodes.
# Validation: Each table node in X/Y direction must be >= BP dimensions
TABLE_DISCRETIZATION_MODE = 1   # 0 = single node, 1+ = subdivided
N_TABLE_X = 3               # Base subdivisions along length (X) for Mode 1
N_TABLE_Y = 2               # Base subdivisions along width (Y) for Mode 1
N_TABLE_Z = 1               # Base subdivisions along thickness (Z) for Mode 1

# --- Interaction & Environment ---
AMBIENT_TEMP = 25.0        # [°C]
CONTACT_BP_TABLE = BP_LENGTH * BP_WIDTH * 0.9    # [m^2] Area (Factor 0.9 for holes in Table)
ALPHA_CONTACT = 600.0      # [W/(m^2 K)] Heat transfer coefficient contact between Base Plate and Table (Gap Conductance)
STEFAN_BOLTZMANN = 5.67e-8 # [W/(m^2 K^4)]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#initialize materials
_material_waam = get_material(MATERIAL_WAAM_NAME)
_material_bp = get_material(MATERIAL_BP_NAME)
_material_table = get_material(MATERIAL_TABLE_NAME)


def get_cp_waam(temp_c):
    """
    Calculates the specific heat capacity for WAAM material.
    """
    return _material_waam.get_cp(temp_c)

def get_cp_bp(temp_c):
    """
    Calculates the specific heat capacity for the base plate.
    """
    return _material_bp.get_cp(temp_c)

def get_cp_table(temp_c):
    """
    Calculates the specific heat capacity for the table.
    """
    return _material_table.get_cp(temp_c)

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
    return _material_waam.get_cp(temps)

def get_cp_table_vectorized(temps):
    """Vectorized version of get_cp_table for numpy arrays."""
    return _material_table.get_cp(temps)

def get_epsilon_waam_vectorized(temps):
    """Vectorized version of get_epsilon_waam for numpy arrays."""
    temps = np.atleast_1d(temps)
    result = np.where(temps >= MELTING_TEMP, EPSILON_WAAM_LIQUID, EPSILON_WAAM)
    return result

# =============================================================================
# EFFICIENT DATA STRUCTURES FOR SIMULATION
# =============================================================================

# Node Types
TYPE_INACTIVE = 0
TYPE_LAYER = 1
TYPE_BEAD = 2
TYPE_ELEMENT = 3
TYPE_SPECIAL = 4  # Base Plate
TYPE_TABLE = 5

class NodeMatrix:
    """
    Matrix-based node tracking for layers, beads, and elements.
    Uses pre-allocated numpy arrays for performance.
    """
    
    def __init__(self, max_waam_nodes, num_table_nodes, num_tracks, num_elements_per_bead):
        # Total nodes = Table + BP (1) + WAAM
        self.num_table_nodes = num_table_nodes
        self.bp_idx = num_table_nodes
        self.waam_start_idx = num_table_nodes + 1
        self.max_nodes = self.waam_start_idx + max_waam_nodes
        
        self.num_tracks = num_tracks
        self.num_elements_per_bead = num_elements_per_bead
        self.nodes_per_bead = num_elements_per_bead
        self.nodes_per_layer = num_tracks * num_elements_per_bead
        
        # Pre-allocate arrays
        self.layer_idx = np.full(self.max_nodes, -1, dtype=np.int32)
        self.bead_idx = np.full(self.max_nodes, -1, dtype=np.int32)
        self.element_idx = np.full(self.max_nodes, -1, dtype=np.int32)
        self.level_type = np.full(self.max_nodes, TYPE_INACTIVE, dtype=np.int8)
        
        self.masses = np.zeros(self.max_nodes, dtype=np.float64)
        self.areas = np.zeros(self.max_nodes, dtype=np.float64)
        self.temperatures = np.full(self.max_nodes, AMBIENT_TEMP, dtype=np.float64)
        self.active_mask = np.zeros(self.max_nodes, dtype=bool)
        self.radiation_areas = np.zeros(self.max_nodes, dtype=np.float64)
        
        # Table nodes (can be single or multiple)
        self.table_indices = []  # List of all table node indices
        self.table_idx = None    # For backwards compatibility (first table node or single table)
        self.table_bp_contact_idx = None  # Table node that contacts baseplate
        
        # Table grid properties (for multi-node table)
        self.table_grid = None   # 3D array of node indices: [ix, iy, iz]
        self.table_node_dims = None  # (dx, dy, dz) dimensions of each table node
        self.table_subdivisions = None  # (nx, ny, nz) number of subdivisions
        
        # Optimization caches
        self.active_waam_indices = set()
        self.bp_covered_area = 0.0
        
    def get_waam_node_idx(self, layer, bead, element):
        """Map (layer, bead, element) to unique index."""
        # If element is -1 (bead level), use element=0
        # If bead is -1 (layer level), use bead=0, element=0
        b = max(0, bead)
        e = max(0, element)
        return self.waam_start_idx + layer * self.nodes_per_layer + b * self.nodes_per_bead + e

    def activate_waam_node(self, layer_idx, bead_idx, element_idx, level_type_str, mass, area, temperature):
        """Activate a WAAM node at the mapped index."""
        idx = self.get_waam_node_idx(layer_idx, bead_idx, element_idx)
        
        # Convert string type to int
        if level_type_str == 'layer': l_type = TYPE_LAYER
        elif level_type_str == 'bead': l_type = TYPE_BEAD
        elif level_type_str == 'element': l_type = TYPE_ELEMENT
        else: l_type = TYPE_INACTIVE
            
        self.layer_idx[idx] = layer_idx
        self.bead_idx[idx] = bead_idx
        self.element_idx[idx] = element_idx
        self.level_type[idx] = l_type
        self.masses[idx] = mass
        self.areas[idx] = area
        self.temperatures[idx] = temperature
        self.active_mask[idx] = True
        
        self.active_waam_indices.add(idx)
        
        # Update BP covered area if layer 0
        if layer_idx == 0:
            self.bp_covered_area += area
            
        self.radiation_areas[idx] = self.calculate_self_radiation(idx)
        self.update_neighbor_radiation(idx, is_activating=True)
        return idx

    def calculate_self_radiation(self, idx):
        """Calculate exposed radiation area for a node based on current neighbors."""
        l_type = self.level_type[idx]
        layer = self.layer_idx[idx]
        bead = self.bead_idx[idx]
        
        # 1. Calculate full exposed area (assuming no neighbors)
        rad_area = 0.0
        
        if l_type == TYPE_ELEMENT:
            element_length = TRACK_LENGTH / self.num_elements_per_bead
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * TRACK_OVERLAP
            rad_area = (element_length * bead_width) + (2 * element_length * LAYER_HEIGHT) + (2 * bead_width * LAYER_HEIGHT)
            
        elif l_type == TYPE_BEAD:
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * TRACK_OVERLAP
            rad_area = (bead_width * TRACK_LENGTH) + (2 * TRACK_LENGTH * LAYER_HEIGHT) + (2 * bead_width * LAYER_HEIGHT)
            
        elif l_type == TYPE_LAYER:
            effective_layer_width = TRACK_WIDTH * self.num_tracks - TRACK_WIDTH * (self.num_tracks - 1) * (1 - TRACK_OVERLAP)
            rad_area = (effective_layer_width * TRACK_LENGTH) + (2 * (TRACK_LENGTH + effective_layer_width) * LAYER_HEIGHT)
            
        # 2. Subtract contact areas of existing neighbors
        # Vertical Down
        if self.get_vertical_neighbor(idx, -1) is not None:
            rad_area -= self.areas[idx]
            
        # Vertical Up
        if self.get_vertical_neighbor(idx, +1) is not None:
            rad_area -= self.areas[idx] # Assuming same area
            
        # Horizontal
        if l_type in (TYPE_BEAD, TYPE_ELEMENT):
            contact = (TRACK_LENGTH if l_type == TYPE_BEAD else TRACK_LENGTH / self.num_elements_per_bead) * LAYER_HEIGHT
            if self.get_horizontal_neighbor(idx, -1) is not None: rad_area -= contact
            if self.get_horizontal_neighbor(idx, +1) is not None: rad_area -= contact
                
        # Longitudinal
        if l_type == TYPE_ELEMENT:
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            contact = bead_width * LAYER_HEIGHT
            if self.get_longitudinal_neighbor(idx, -1) is not None: rad_area -= contact
            if self.get_longitudinal_neighbor(idx, +1) is not None: rad_area -= contact
        
        return max(0.0, rad_area)

    def update_neighbor_radiation(self, idx, is_activating=True):
        """Update neighbors' radiation area when this node is activated/deactivated."""
        l_type = self.level_type[idx]
        
        def update(neighbor_idx, contact_area):
            if neighbor_idx is None: return
            if is_activating:
                self.radiation_areas[neighbor_idx] = max(0.0, self.radiation_areas[neighbor_idx] - contact_area)
            else:
                self.radiation_areas[neighbor_idx] += contact_area

        # Vertical Down
        update(self.get_vertical_neighbor(idx, -1), self.areas[idx])
        
        # Vertical Up
        update(self.get_vertical_neighbor(idx, +1), self.areas[idx])
            
        # Horizontal
        if l_type in (TYPE_BEAD, TYPE_ELEMENT):
            contact = (TRACK_LENGTH if l_type == TYPE_BEAD else TRACK_LENGTH / self.num_elements_per_bead) * LAYER_HEIGHT
            update(self.get_horizontal_neighbor(idx, -1), contact)
            update(self.get_horizontal_neighbor(idx, +1), contact)
                
        # Longitudinal
        if l_type == TYPE_ELEMENT:
            bead = self.bead_idx[idx]
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * (1 - TRACK_OVERLAP)
            contact = bead_width * LAYER_HEIGHT
            update(self.get_longitudinal_neighbor(idx, -1), contact)
            update(self.get_longitudinal_neighbor(idx, +1), contact)

    def add_table_node(self, idx, mass, area, temperature):
        """Activate a table node."""
        # idx must be < num_table_nodes
        self.level_type[idx] = TYPE_TABLE
        self.masses[idx] = mass
        self.areas[idx] = area
        self.temperatures[idx] = temperature
        self.active_mask[idx] = True
        self.table_indices.append(idx)
        return idx

    def add_bp_node(self, mass, area, temperature):
        """Activate the base plate node."""
        idx = self.bp_idx
        self.level_type[idx] = TYPE_SPECIAL
        self.masses[idx] = mass
        self.areas[idx] = area
        self.temperatures[idx] = temperature
        self.active_mask[idx] = True
        return idx
    
    def get_top_layer_idx(self):
        """Get the layer index of the topmost layer."""
        # Find max layer_idx where active and not special/table
        # Optimization: check from end of active WAAM nodes
        # Or just track it externally?
        # Vectorized search:
        waam_indices = np.where(self.active_mask[self.waam_start_idx:])[0]
        if len(waam_indices) == 0:
            return -1
        # Map back to global index
        global_indices = waam_indices + self.waam_start_idx
        return np.max(self.layer_idx[global_indices])
    
    def get_nodes_in_layer(self, layer_idx):
        """Get all active node indices in a specific layer."""
        start = self.waam_start_idx + layer_idx * self.nodes_per_layer
        end = start + self.nodes_per_layer
        # Check bounds
        if start >= self.max_nodes: return []
        
        indices = np.arange(start, min(end, self.max_nodes))
        return indices[self.active_mask[indices]].tolist()
    
    def get_nodes_in_bead(self, layer_idx, bead_idx):
        """Get all active element nodes within a specific bead."""
        start = self.waam_start_idx + layer_idx * self.nodes_per_layer + bead_idx * self.nodes_per_bead
        end = start + self.nodes_per_bead
        
        indices = np.arange(start, end)
        # Filter for active AND element type (though if active in this range it should be element)
        mask = self.active_mask[indices] & (self.level_type[indices] == TYPE_ELEMENT)
        return indices[mask].tolist()
    
    def get_beads_in_layer(self, layer_idx):
        """Get list of bead indices that exist in a layer."""
        # This is used for consolidation.
        # We can check the representative nodes for each bead
        bead_indices = []
        for b in range(self.num_tracks):
            # Check if any node in this bead range is active
            start = self.waam_start_idx + layer_idx * self.nodes_per_layer + b * self.nodes_per_bead
            end = start + self.nodes_per_bead
            if np.any(self.active_mask[start:end]):
                bead_indices.append(b)
        return bead_indices

    def get_layer_level_type(self, layer_idx):
        """Get the discretization level type for a specific layer."""
        # Check the first active node in the layer
        nodes = self.get_nodes_in_layer(layer_idx)
        if not nodes:
            return None
        
        l_type = self.level_type[nodes[0]]
        if l_type == TYPE_LAYER: return 'layer'
        if l_type == TYPE_BEAD: return 'bead'
        if l_type == TYPE_ELEMENT: return 'element'
        return None
    
    def get_vertical_neighbor(self, node_idx, direction):
        """
        Get vertical neighbor (±1 in layer direction).
        Returns node index or None if not found.
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        l_type = self.level_type[node_idx]
        
        target_layer = layer + direction
        
        # Special case: going below layer 0 leads to base plate
        if target_layer < 0:
            return self.bp_idx
        
        # Check if target layer exists
        if target_layer >= NUMBER_OF_LAYERS: # Assuming NUMBER_OF_LAYERS is global
             return None

        # Get target layer type
        # We can check the representative node for the target layer
        # Or just try to find the matching node
        
        # If same level, we can calculate index directly
        target_idx = self.get_waam_node_idx(target_layer, bead, element)
        
        if self.active_mask[target_idx] and self.level_type[target_idx] == l_type:
            return target_idx
            
        return None
    
    def get_cross_level_vertical_info(self, node_idx, direction):
        """
        Get information for cross-level vertical heat transfer.
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        l_type = self.level_type[node_idx]
        
        target_layer = layer + direction
        
        # Special case: base plate
        if target_layer < 0:
            return {'type': 'to_baseplate', 'target_nodes': [self.bp_idx], 
                    'source_nodes': [node_idx], 'target_layer': -1, 'target_level': 'special'}
        
        # Get target layer's level type
        target_level_str = self.get_layer_level_type(target_layer)
        if target_level_str is None:
             return {'type': 'none', 'target_nodes': [], 'source_nodes': [], 
                    'target_layer': target_layer, 'target_level': None}
        
        target_level = {'layer': TYPE_LAYER, 'bead': TYPE_BEAD, 'element': TYPE_ELEMENT}[target_level_str]

        # Same level - direct transfer
        if l_type == target_level:
            neighbor = self.get_vertical_neighbor(node_idx, direction)
            if neighbor is not None:
                return {'type': 'same', 'target_nodes': [neighbor], 
                        'source_nodes': [node_idx], 'target_layer': target_layer, 
                        'target_level': target_level_str}
            return {'type': 'none', 'target_nodes': [], 'source_nodes': [], 
                    'target_layer': target_layer, 'target_level': target_level_str}
        
        # Element → Bead (finer to coarser)
        if l_type == TYPE_ELEMENT and target_level == TYPE_BEAD:
            # Target is the bead node
            target_idx = self.get_waam_node_idx(target_layer, bead, -1) # Element -1 maps to 0
            if self.active_mask[target_idx]:
                return {'type': 'element_to_bead', 'target_nodes': [target_idx], 
                        'source_nodes': [node_idx], 'target_layer': target_layer, 'target_level': 'bead'}
        
        # Bead → Element (coarser to finer)
        if l_type == TYPE_BEAD and target_level == TYPE_ELEMENT:
            # Target are all elements in the bead
            target_nodes = self.get_nodes_in_bead(target_layer, bead)
            return {'type': 'bead_to_element', 'target_nodes': target_nodes, 
                    'source_nodes': [node_idx], 'target_layer': target_layer, 'target_level': 'element'}
            
        # Bead → Layer
        if l_type == TYPE_BEAD and target_level == TYPE_LAYER:
            target_idx = self.get_waam_node_idx(target_layer, -1, -1)
            if self.active_mask[target_idx]:
                return {'type': 'bead_to_layer', 'target_nodes': [target_idx], 
                        'source_nodes': [node_idx], 'target_layer': target_layer, 'target_level': 'layer'}

        # Layer → Bead
        if l_type == TYPE_LAYER and target_level == TYPE_BEAD:
            # Target are all beads in layer
            # We need to find which beads are active in target layer
            # But wait, layer node covers ALL beads.
            # So it connects to ALL beads in target layer.
            target_nodes = self.get_nodes_in_layer(target_layer) # Should be all beads
            return {'type': 'layer_to_bead', 'target_nodes': target_nodes, 
                    'source_nodes': [node_idx], 'target_layer': target_layer, 'target_level': 'bead'}

        return {'type': 'none', 'target_nodes': [], 'source_nodes': [], 
                'target_layer': target_layer, 'target_level': target_level_str} 


    
    def compute_mass_weighted_temperature(self, node_indices, temperatures):
        """Compute mass-weighted average temperature for a list of nodes."""
        if not node_indices:
            return 0.0
        # Vectorized calculation
        indices = np.array(node_indices)
        masses = self.masses[indices]
        temps = temperatures[indices]
        total_mass = np.sum(masses)
        if total_mass == 0:
            return temps[0] if len(temps) > 0 else 0.0
        return np.sum(masses * temps) / total_mass
    
    def get_horizontal_neighbor(self, node_idx, direction):
        """
        Get horizontal neighbor (±1 in bead direction).
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        l_type = self.level_type[node_idx]
        
        if bead == -1: return None
        
        target_bead = bead + direction
        if target_bead < 0 or target_bead >= self.num_tracks: return None
        
        target_idx = self.get_waam_node_idx(layer, target_bead, element)
        if self.active_mask[target_idx] and self.level_type[target_idx] == l_type:
            return target_idx
        return None
    
    def get_longitudinal_neighbor(self, node_idx, direction):
        """
        Get longitudinal neighbor (±1 in element direction along bead).
        """
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        l_type = self.level_type[node_idx]
        
        if element == -1 or l_type != TYPE_ELEMENT: return None
        
        target_element = element + direction
        if target_element < 0 or target_element >= self.num_elements_per_bead: return None
        
        target_idx = self.get_waam_node_idx(layer, bead, target_element)
        if self.active_mask[target_idx] and self.level_type[target_idx] == l_type:
            return target_idx
        return None
    
    def get_elements_placed_in_bead(self, layer_idx, bead_idx):
        """Get the number of elements already placed in a specific bead position."""
        # Check active elements in this bead
        start = self.waam_start_idx + layer_idx * self.nodes_per_layer + bead_idx * self.nodes_per_bead
        end = start + self.nodes_per_bead
        # Count active elements
        return np.sum(self.active_mask[start:end] & (self.level_type[start:end] == TYPE_ELEMENT))
    
    def get_max_bead_in_layer(self, layer_idx):
        """Get the maximum bead index currently existing in a layer."""
        beads = self.get_beads_in_layer(layer_idx)
        return max(beads) if beads else -1
    
    def get_max_element_in_bead(self, layer_idx, bead_idx):
        """Get the maximum element index currently existing in a bead."""
        start = self.waam_start_idx + layer_idx * self.nodes_per_layer + bead_idx * self.nodes_per_bead
        end = start + self.nodes_per_bead
        # Find last active element
        active_indices = np.where(self.active_mask[start:end] & (self.level_type[start:end] == TYPE_ELEMENT))[0]
        return active_indices[-1] if len(active_indices) > 0 else -1
    
    # === Table Grid Methods ===
    
    def initialize_table_grid(self, nx, ny, nz, table_length, table_width, table_thickness, 
                               rho_table, temp_init):
        """
        Initialize the table as a 3D grid of thermal nodes.
        
        Args:
            nx, ny, nz: Number of subdivisions in length (X), width (Y), thickness (Z)
            table_length, table_width, table_thickness: Table dimensions [m]
            rho_table: Table density [kg/m³]
            temp_init: Initial temperature [°C]
        
        Returns:
            3D numpy array of node indices
        """
        # Calculate node dimensions
        dx = table_length / nx
        dy = table_width / ny
        dz = table_thickness / nz
        
        self.table_node_dims = (dx, dy, dz)
        self.table_subdivisions = (nx, ny, nz)
        
        # Calculate node volume and mass
        node_volume = dx * dy * dz
        node_mass = node_volume * rho_table
        
        # Create 3D grid array for node indices
        self.table_grid = np.zeros((nx, ny, nz), dtype=np.int32)
        self.table_indices = []
        
        # Add nodes for each grid cell
        idx_counter = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    node_idx = idx_counter
                    self.add_table_node(
                        node_idx,
                        mass=node_mass, 
                        area=dx * dy,
                        temperature=temp_init
                    )
                    self.bead_idx[node_idx] = ix * 100 + iy
                    self.element_idx[node_idx] = iz
                    self.table_grid[ix, iy, iz] = node_idx
                    idx_counter += 1
        
        self.table_idx = self.table_indices[0]
        return self.table_grid
    
    def get_table_grid_position(self, node_idx):
        """Get the (ix, iy, iz) grid position of a table node."""
        if self.table_grid is None: return None
        if self.level_type[node_idx] != TYPE_TABLE: return None
        encoded = self.bead_idx[node_idx]
        iz = self.element_idx[node_idx]
        ix = encoded // 100
        iy = encoded % 100
        return (ix, iy, iz)
    
    def get_table_neighbors(self, node_idx):
        """Get neighboring table nodes."""
        if self.table_grid is None: return {}
        pos = self.get_table_grid_position(node_idx)
        if pos is None: return {}
        
        ix, iy, iz = pos
        nx, ny, nz = self.table_subdivisions
        
        neighbors = {}
        neighbors['x+'] = self.table_grid[ix+1, iy, iz] if ix + 1 < nx else None
        neighbors['x-'] = self.table_grid[ix-1, iy, iz] if ix > 0 else None
        neighbors['y+'] = self.table_grid[ix, iy+1, iz] if iy + 1 < ny else None
        neighbors['y-'] = self.table_grid[ix, iy-1, iz] if iy > 0 else None
        neighbors['z+'] = self.table_grid[ix, iy, iz+1] if iz + 1 < nz else None
        neighbors['z-'] = self.table_grid[ix, iy, iz-1] if iz > 0 else None
        return neighbors
    
    def compute_table_node_radiation_area(self, node_idx):
        """Compute radiation area for a table node."""
        if self.table_grid is None: return 0.0
        pos = self.get_table_grid_position(node_idx)
        if pos is None: return 0.0
        
        ix, iy, iz = pos
        nx, ny, nz = self.table_subdivisions
        dx, dy, dz = self.table_node_dims
        
        rad_area = 0.0
        if ix == 0: rad_area += dy * dz
        if ix == nx - 1: rad_area += dy * dz
        if iy == 0: rad_area += dx * dz
        if iy == ny - 1: rad_area += dx * dz
        if iz == 0: rad_area += dx * dy
        if iz == nz - 1:
            if node_idx == self.table_bp_contact_idx:
                top_area = dx * dy - CONTACT_BP_TABLE
                if top_area > 0: rad_area += top_area
            else:
                rad_area += dx * dy
        return rad_area

    def compute_radiation_area(self, node_idx, top_layer_idx):
        """Compute radiation area for a node."""
        # Note: active_mask check is done by caller or implicit in active_waam_indices
        
        level = self.level_type[node_idx]
        if level == TYPE_SPECIAL: return 0.0
        
        layer = self.layer_idx[node_idx]
        bead = self.bead_idx[node_idx]
        element = self.element_idx[node_idx]
        
        rad_area = 0.0
        is_top = (layer == top_layer_idx)
        
        effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
        
        if level == TYPE_LAYER:
            layer_area = effective_layer_width * TRACK_LENGTH
            side_area = 2 * (TRACK_LENGTH + effective_layer_width) * LAYER_HEIGHT
            if is_top: rad_area += layer_area
            rad_area += side_area
            
        elif level == TYPE_BEAD:
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * TRACK_OVERLAP
            bead_top_area = bead_width * TRACK_LENGTH
            
            if is_top: rad_area += bead_top_area
            
            # Side surfaces (Long sides)
            # Check neighbors to determine if exposed
            if self.get_horizontal_neighbor(node_idx, -1) is None:
                rad_area += TRACK_LENGTH * LAYER_HEIGHT
            if self.get_horizontal_neighbor(node_idx, +1) is None:
                rad_area += TRACK_LENGTH * LAYER_HEIGHT
            
            # Short sides (always exposed for beads)
            rad_area += 2 * bead_width * LAYER_HEIGHT
            
            # Top layer missing neighbors check is covered by the above logic?
            # Original logic: "In top layer: check for missing horizontal neighbors"
            # If I am in top layer, and neighbor is missing, I radiate.
            # My new logic: If neighbor is missing (regardless of layer), I radiate.
            # This is physically correct. A side face is exposed if there is no neighbor.
            # The only difference is if a neighbor exists but is NOT active?
            # get_horizontal_neighbor checks active_mask. So if neighbor is not active, it returns None.
            # So this covers all cases.
                    
        elif level == TYPE_ELEMENT:
            element_length = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * TRACK_OVERLAP
            element_top_area = element_length * bead_width
            
            if is_top: rad_area += element_top_area
            
            # Broad sides (perpendicular to track direction)
            # Check longitudinal neighbors
            if self.get_longitudinal_neighbor(node_idx, -1) is None:
                rad_area += bead_width * LAYER_HEIGHT
            if self.get_longitudinal_neighbor(node_idx, +1) is None:
                rad_area += bead_width * LAYER_HEIGHT
            
            # Long sides (parallel to track direction)
            # Check horizontal neighbors (bead neighbors)
            # Note: get_horizontal_neighbor works for elements too (checks same element index in neighbor bead)
            # But wait, if neighbor bead exists but is shorter (elements not yet placed), 
            # get_horizontal_neighbor might return None.
            # This is correct: if the neighbor element doesn't exist, my side is exposed.
            
            if self.get_horizontal_neighbor(node_idx, -1) is None:
                rad_area += element_length * LAYER_HEIGHT
            if self.get_horizontal_neighbor(node_idx, +1) is None:
                rad_area += element_length * LAYER_HEIGHT
                    
        return rad_area

    def compute_bp_radiation_area(self, top_layer_idx):
        """Compute base plate radiation area."""
        if top_layer_idx < 0:
            return (2 * BP_LENGTH * BP_WIDTH + 
                    2 * BP_LENGTH * BP_THICKNESS + 
                    2 * BP_WIDTH * BP_THICKNESS)
        
        # Use cached covered area
        covered_area = self.bp_covered_area
        
        if top_layer_idx == 0:
            bp_top_uncovered = max(0, BP_LENGTH * BP_WIDTH - covered_area)
            bp_sides = 2 * BP_LENGTH * BP_THICKNESS + 2 * BP_WIDTH * BP_THICKNESS
            return bp_top_uncovered + bp_sides
        else:
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

    def consolidate_layer_to_beads(self, layer_idx):
        """Consolidate all elements in a layer into beads."""
        # Find all element nodes in this layer
        layer_mask = (self.layer_idx == layer_idx) & (self.level_type == TYPE_ELEMENT) & self.active_mask
        element_indices = np.where(layer_mask)[0]
        
        if len(element_indices) == 0: return
        
        # Group by bead
        beads_in_layer = np.unique(self.bead_idx[element_indices])
        
        for bead in beads_in_layer:
            # Find elements for this bead
            bead_mask = layer_mask & (self.bead_idx == bead)
            bead_elements = np.where(bead_mask)[0]
            
            if len(bead_elements) == 0: continue
            
            # Calculate average temperature weighted by mass
            total_mass = np.sum(self.masses[bead_elements])
            weighted_temp = np.sum(self.temperatures[bead_elements] * self.masses[bead_elements])
            avg_temp = weighted_temp / total_mass if total_mass > 0 else AMBIENT_TEMP
            
            # Deactivate elements
            for idx in bead_elements:
                self.update_neighbor_radiation(idx, is_activating=False)
                self.active_mask[idx] = False
                self.active_waam_indices.discard(idx)
                if layer_idx == 0:
                    self.bp_covered_area -= self.areas[idx]
            
            # Activate bead node (use first element slot)
            bead_node_idx = self.get_waam_node_idx(layer_idx, bead, 0)
            
            # Update properties
            self.level_type[bead_node_idx] = TYPE_BEAD
            self.active_mask[bead_node_idx] = True
            self.masses[bead_node_idx] = total_mass
            self.temperatures[bead_node_idx] = avg_temp
            
            # Area calculation for bead
            bead_width = TRACK_WIDTH if bead == 0 else TRACK_WIDTH * TRACK_OVERLAP
            self.areas[bead_node_idx] = bead_width * TRACK_LENGTH
            
            self.active_waam_indices.add(bead_node_idx)
            if layer_idx == 0:
                self.bp_covered_area += self.areas[bead_node_idx]
                
            # Update radiation
            self.radiation_areas[bead_node_idx] = self.calculate_self_radiation(bead_node_idx)
            self.update_neighbor_radiation(bead_node_idx, is_activating=True)

    def consolidate_layer_to_layer(self, layer_idx):
        """Consolidate all beads/elements in a layer into a single layer node."""
        # Find all active nodes in this layer (beads or elements)
        layer_mask = (self.layer_idx == layer_idx) & self.active_mask & (self.level_type != TYPE_LAYER)
        nodes_in_layer = np.where(layer_mask)[0]
        
        if len(nodes_in_layer) == 0: return
        
        # Calculate average temperature weighted by mass
        total_mass = np.sum(self.masses[nodes_in_layer])
        weighted_temp = np.sum(self.temperatures[nodes_in_layer] * self.masses[nodes_in_layer])
        avg_temp = weighted_temp / total_mass if total_mass > 0 else AMBIENT_TEMP
        
        # Deactivate all current nodes
        for idx in nodes_in_layer:
            self.update_neighbor_radiation(idx, is_activating=False)
            self.active_mask[idx] = False
            self.active_waam_indices.discard(idx)
            if layer_idx == 0:
                self.bp_covered_area -= self.areas[idx]
        
        # Activate layer node (use first slot of first bead)
        layer_node_idx = self.get_waam_node_idx(layer_idx, 0, 0)
        
        self.level_type[layer_node_idx] = TYPE_LAYER
        self.active_mask[layer_node_idx] = True
        self.masses[layer_node_idx] = total_mass
        self.temperatures[layer_node_idx] = avg_temp
        
        # Area calculation for layer
        effective_layer_width = TRACK_WIDTH * NUMBER_OF_TRACKS - TRACK_WIDTH * (NUMBER_OF_TRACKS - 1) * (1 - TRACK_OVERLAP)
        self.areas[layer_node_idx] = effective_layer_width * TRACK_LENGTH
        
        self.active_waam_indices.add(layer_node_idx)
        if layer_idx == 0:
            self.bp_covered_area += self.areas[layer_node_idx]
            
        # Update radiation
        self.radiation_areas[layer_node_idx] = self.calculate_self_radiation(layer_node_idx)
        self.update_neighbor_radiation(layer_node_idx, is_activating=True)


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


def update_temperatures_matrix(node_matrix, model, dt, is_welding=False, arc_power=0.0, welding_node_idx=None):
    """
    Update temperatures using NodeMatrix for efficient neighbor finding and radiation calculation.
    """
    num_nodes = len(node_matrix.temperatures)
    Q_balance = np.zeros(num_nodes, dtype=np.float64)
    
    # Get temperature array (view)
    T = node_matrix.temperatures
    
    # Pre-convert active indices to array for vectorization
    active_waam_indices_list = list(node_matrix.active_waam_indices)
    active_indices_arr = np.array(active_waam_indices_list, dtype=np.int32) if active_waam_indices_list else np.array([], dtype=np.int32)
    
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
            
            # Fix: Use correct length for contact area (Element vs Bead)
            if node_matrix.level_type[welding_node_idx] == TYPE_ELEMENT:
                length = TRACK_LENGTH / node_matrix.num_elements_per_bead
            else:
                length = TRACK_LENGTH
                
            # Contact area for arc power distribution: length * height (vertical face)
            contact_areas.append(length * LAYER_HEIGHT)
        
        # Vertical neighbor (layer below)
        v_neighbor = node_matrix.get_vertical_neighbor(welding_node_idx, -1)
        if v_neighbor is not None:
            neighbors.append(v_neighbor)
            contact_areas.append(node_matrix.areas[v_neighbor])
        elif node_matrix.bp_idx is not None:
            neighbors.append(node_matrix.bp_idx)
            contact_areas.append(node_matrix.areas[welding_node_idx])
        
        # Special case: first node in layer
        is_first_node = (node_matrix.bead_idx[welding_node_idx] == 0 and 
                        node_matrix.get_horizontal_neighbor(welding_node_idx, -1) is None)
        
        if is_first_node and node_matrix.bp_idx in neighbors:
            bp_idx = neighbors.index(node_matrix.bp_idx)
            Q_balance[node_matrix.bp_idx] += remaining_power
        elif neighbors:
            total_contact_area = sum(contact_areas)
            if total_contact_area > 0:
                for i, neighbor_idx in enumerate(neighbors):
                    fraction = contact_areas[i] / total_contact_area
                    Q_balance[neighbor_idx] += remaining_power * fraction
            else:
                equal_share = remaining_power / len(neighbors)
                for neighbor_idx in neighbors:
                    Q_balance[neighbor_idx] += equal_share
    
    # --- Radiation ---
    T_K4 = (T + 273.15)**4
    T_amb_K4 = (AMBIENT_TEMP + 273.15)**4
    top_layer_idx = node_matrix.get_top_layer_idx()
    
    # Table radiation
    if node_matrix.table_grid is not None:
        for table_node_idx in node_matrix.table_indices:
            table_rad_area = node_matrix.compute_table_node_radiation_area(table_node_idx)
            if table_rad_area > 0:
                Q_balance[table_node_idx] -= (EPSILON_TABLE * STEFAN_BOLTZMANN * 
                                              table_rad_area * 
                                              (T_K4[table_node_idx] - T_amb_K4))
    elif node_matrix.table_idx is not None:
        table_rad_area = node_matrix.compute_table_radiation_area()
        Q_balance[node_matrix.table_idx] -= (EPSILON_TABLE * STEFAN_BOLTZMANN * 
                                              table_rad_area * 
                                              (T_K4[node_matrix.table_idx] - T_amb_K4))
    
    # Base plate radiation
    if node_matrix.bp_idx is not None:
        bp_rad_area = node_matrix.compute_bp_radiation_area(top_layer_idx)
        Q_balance[node_matrix.bp_idx] -= (EPSILON_BP * STEFAN_BOLTZMANN * 
                                          bp_rad_area * 
                                          (T_K4[node_matrix.bp_idx] - T_amb_K4))
    
    # WAAM nodes radiation
    # Use cached active indices and pre-calculated radiation areas
    if len(active_indices_arr) > 0:
        T_active = T[active_indices_arr]
        
        # Vectorized properties
        epsilon_active = get_epsilon_waam_vectorized(T_active)
        rad_areas_active = node_matrix.radiation_areas[active_indices_arr]
        
        # Calculate radiation
        Q_rad = epsilon_active * STEFAN_BOLTZMANN * rad_areas_active * (T_K4[active_indices_arr] - T_amb_K4)
        Q_balance[active_indices_arr] -= Q_rad
    
    # --- Conduction ---
    # Table internal conduction
    if node_matrix.table_grid is not None:
        dx, dy, dz = node_matrix.table_node_dims
        processed_table_pairs = set()
        
        for table_node_idx in node_matrix.table_indices:
            neighbors = node_matrix.get_table_neighbors(table_node_idx)
            for direction, neighbor_idx in neighbors.items():
                if neighbor_idx is None: continue
                pair_key = (min(table_node_idx, neighbor_idx), max(table_node_idx, neighbor_idx))
                if pair_key in processed_table_pairs: continue
                processed_table_pairs.add(pair_key)
                
                if direction in ('x+', 'x-'): area, dist = dy * dz, dx
                elif direction in ('y+', 'y-'): area, dist = dx * dz, dy
                else: area, dist = dx * dy, dz
                
                q_cond = LAMBDA_TABLE * area / dist * (T[neighbor_idx] - T[table_node_idx])
                Q_balance[table_node_idx] += q_cond
                Q_balance[neighbor_idx] -= q_cond
    
    # Table <-> Base plate conduction
    if node_matrix.table_bp_contact_idx is not None and node_matrix.bp_idx is not None:
        q_contact = ALPHA_CONTACT * CONTACT_BP_TABLE * (T[node_matrix.bp_idx] - T[node_matrix.table_bp_contact_idx])
        Q_balance[node_matrix.table_bp_contact_idx] += q_contact
        Q_balance[node_matrix.bp_idx] -= q_contact
    
    # Conduction between all nodes
    processed_pairs = set()
    
    # BP to first layer
    if node_matrix.bp_idx is not None:
        bp_idx = node_matrix.bp_idx
        first_layer_nodes = node_matrix.get_nodes_in_layer(0)
        if first_layer_nodes:
            dist = (BP_THICKNESS / 2) + (LAYER_HEIGHT / 2)
            lam = (LAMBDA_BP + LAMBDA_WAAM) / 2
            for target_idx in first_layer_nodes:
                pair_key = (min(bp_idx, target_idx), max(bp_idx, target_idx))
                if pair_key in processed_pairs: continue
                processed_pairs.add(pair_key)
                
                area = node_matrix.areas[target_idx]
                q_vert = lam * area / dist * (T[target_idx] - T[bp_idx])
                Q_balance[bp_idx] += q_vert
                Q_balance[target_idx] -= q_vert
    
    # WAAM nodes conduction
    for i in active_waam_indices_list:
        # Vertical conduction
        cross_info = node_matrix.get_cross_level_vertical_info(i, +1)
        transfer_type = cross_info['type']
        
        if transfer_type == 'to_baseplate': pass
        elif transfer_type == 'none' or transfer_type is None: pass
        elif transfer_type == 'same':
            v_up = cross_info['target_nodes'][0]
            pair_key = (min(i, v_up), max(i, v_up))
            if pair_key not in processed_pairs:
                processed_pairs.add(pair_key)
                dist = LAYER_HEIGHT
                lam = LAMBDA_WAAM
                area = node_matrix.areas[v_up]
                q_vert = lam * area / dist * (T[v_up] - T[i])
                Q_balance[i] += q_vert
                Q_balance[v_up] -= q_vert
        elif transfer_type in ['element_to_bead', 'bead_to_layer', 'element_to_layer']:
            target_nodes = cross_info['target_nodes']
            if target_nodes:
                target_idx = target_nodes[0]
                dist = LAYER_HEIGHT
                lam = LAMBDA_WAAM
                area = node_matrix.areas[i]
                q_vert = lam * area / dist * (T[target_idx] - T[i])
                Q_balance[i] += q_vert
                Q_balance[target_idx] -= q_vert
        elif transfer_type in ['bead_to_element', 'layer_to_bead', 'layer_to_element']:
            target_nodes = cross_info['target_nodes']
            if target_nodes:
                for target_idx in target_nodes:
                    pair_key = (min(i, target_idx), max(i, target_idx))
                    if pair_key not in processed_pairs:
                        processed_pairs.add(pair_key)
                        dist = LAYER_HEIGHT
                        lam = LAMBDA_WAAM
                        area = node_matrix.areas[target_idx]
                        q_vert = lam * area / dist * (T[target_idx] - T[i])
                        Q_balance[i] += q_vert
                        Q_balance[target_idx] -= q_vert
        
        # Horizontal conduction
        h_right = node_matrix.get_horizontal_neighbor(i, +1)
        if h_right is not None:
            # Distance for conduction is the center-to-center distance
            dist = TRACK_WIDTH * TRACK_OVERLAP
            
            # Fix: Use correct length for contact area (Element vs Bead)
            if node_matrix.level_type[i] == TYPE_ELEMENT:
                length = TRACK_LENGTH / node_matrix.num_elements_per_bead
            else:
                length = TRACK_LENGTH
                
            # Contact area: perpendicular to heat flow direction (horizontal)
            # Heat flows horizontally, so area = length * LAYER_HEIGHT
            contact_area = length * LAYER_HEIGHT
            q_horiz = LAMBDA_WAAM * contact_area / dist * (T[h_right] - T[i])
            Q_balance[i] += q_horiz
            Q_balance[h_right] -= q_horiz
        
        # Longitudinal conduction
        l_forward = node_matrix.get_longitudinal_neighbor(i, +1)
        if l_forward is not None:
            element_length = TRACK_LENGTH / node_matrix.num_elements_per_bead
            bead_width = (TRACK_WIDTH if node_matrix.bead_idx[i] == 0 else TRACK_WIDTH * TRACK_OVERLAP)
            area = bead_width * LAYER_HEIGHT
            dist = element_length
            q_long = LAMBDA_WAAM * area / dist * (T[l_forward] - T[i])
            Q_balance[i] += q_long
            Q_balance[l_forward] -= q_long
            
    # --- Temperature Update ---
    new_T = T.copy()
    
    # Update all active nodes
    # Use cached active indices for WAAM nodes
    if len(active_indices_arr) > 0:
        T_active = T[active_indices_arr]
        cp_active = get_cp_waam_vectorized(T_active)
        masses_active = node_matrix.masses[active_indices_arr]
        
        # Avoid division by zero if mass is 0 (should not happen for active nodes)
        # But let's be safe
        valid_mass = masses_active > 0
        if np.all(valid_mass):
            new_T[active_indices_arr] += (Q_balance[active_indices_arr] * dt) / (masses_active * cp_active)
        else:
            # Fallback for safe update
            safe_indices = active_indices_arr[valid_mass]
            new_T[safe_indices] += (Q_balance[safe_indices] * dt) / (masses_active[valid_mass] * cp_active[valid_mass])
            
    # Update Table and BP nodes
    if node_matrix.table_grid is not None:
        # Vectorized update for table nodes
        table_indices = np.array(node_matrix.table_indices)
        if len(table_indices) > 0:
            T_table = T[table_indices]
            cp_table = get_cp_table_vectorized(T_table)
            masses_table = node_matrix.masses[table_indices]
            new_T[table_indices] += (Q_balance[table_indices] * dt) / (masses_table * cp_table)
            
    elif node_matrix.table_idx is not None:
        i = node_matrix.table_idx
        cp = get_cp_table(T[i])
        new_T[i] += (Q_balance[i] * dt) / (model.m_table * cp)
        
    if node_matrix.bp_idx is not None:
        i = node_matrix.bp_idx
        cp = get_cp_bp(T[i])
        new_T[i] += (Q_balance[i] * dt) / (model.m_bp * cp)
            
    node_matrix.temperatures[:] = new_T
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
    
    # Numerical integration of ∫cp dT from ambient_temp to melting_temp
    # Using trapezoidal rule on discretized array to avoid quad integration warnings
    # with non-smooth interpolated Cp functions.
    num_steps = int(max(100, (melting_temp - ambient_temp) * 2)) # Resolution approx 0.5K
    temps = np.linspace(ambient_temp, melting_temp, num_steps)
    cps = get_cp_waam_vectorized(temps)
    energy_per_kg = np.trapezoid(cps, temps)  # [J/kg]
    
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
    
    # --- Stability Check & Auto-Correction ---
    # Calculate thermal diffusivity at room temperature (worst case for conduction)
    cp_ref = get_cp_waam(20.0)
    alpha = LAMBDA_WAAM / (RHO_WAAM * cp_ref)
    
    # Determine smallest spatial dimension
    # 1. Layer height
    d_z = LAYER_HEIGHT
    # 2. Effective bead width (distance between bead centers)
    d_y = TRACK_WIDTH * TRACK_OVERLAP
    # 3. Element length (if elements are used)
    if N_LAYERS_WITH_ELEMENTS > 0 and N_ELEMENTS_PER_BEAD > 0:
        d_x = TRACK_LENGTH / N_ELEMENTS_PER_BEAD
    else:
        d_x = TRACK_LENGTH # Bead length
        
    # Rigorous 3D Stability criterion: dt <= 1 / (2 * alpha * (1/dx^2 + 1/dy^2 + 1/dz^2))
    inv_sq_sum = (1.0/d_x**2) + (1.0/d_y**2) + (1.0/d_z**2)
    dt_max_stable = 1.0 / (2.0 * alpha * inv_sq_sum)
    
    # Use local variable for simulation time step
    dt_sim = DT
    
    if dt_sim > dt_max_stable:
        print(f"WARNING: Configured DT ({dt_sim:.4f}s) is unstable for the current discretization.")
        print(f"  Smallest dimension: {dx_min*1000:.2f} mm")
        print(f"  Max stable DT: {dt_max_stable:.4f} s")
        print(f"  -> Auto-adjusting DT to {dt_max_stable:.4f} s")
        dt_sim = dt_max_stable
    
    # Calculate geometry
    effective_layer_width = TRACK_WIDTH + (NUMBER_OF_TRACKS - 1) * TRACK_WIDTH * TRACK_OVERLAP
    layer_area = effective_layer_width * TRACK_LENGTH
    layer_volume = layer_area * LAYER_HEIGHT
    side_area_layer = 2 * (TRACK_LENGTH * LAYER_HEIGHT + effective_layer_width * LAYER_HEIGHT)
    
    total_weld_distance = NUMBER_OF_TRACKS * TRACK_LENGTH
    layer_duration = total_weld_distance / PROCESS_SPEED
    
    # Bead geometry
    bead_width_added = TRACK_WIDTH * TRACK_OVERLAP
    bead_area_first = TRACK_WIDTH * TRACK_LENGTH
    bead_area_subsequent = bead_width_added * TRACK_LENGTH
    bead_volume_first = bead_area_first * LAYER_HEIGHT
    bead_volume_subsequent = bead_area_subsequent * LAYER_HEIGHT
    m_bead_first = bead_volume_first * RHO_WAAM
    m_bead_subsequent = bead_volume_subsequent * RHO_WAAM
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
        'bead_duration': bead_duration
    }
    
    thermal_model = ThermalModel(layer_area, side_area_layer, bead_params)
    
    # --- Table discretization calculation ---
    if TABLE_DISCRETIZATION_MODE == 0:
        num_table_nodes = 1
        nx, ny, nz = 1, 1, 1
    else:
        nx = N_TABLE_X + (TABLE_DISCRETIZATION_MODE - 1)
        ny = N_TABLE_Y + (TABLE_DISCRETIZATION_MODE - 1)
        nz = N_TABLE_Z + (TABLE_DISCRETIZATION_MODE - 1)
        num_table_nodes = nx * ny * nz

    # Calculate max WAAM nodes
    max_waam_nodes = NUMBER_OF_LAYERS * NUMBER_OF_TRACKS * N_ELEMENTS_PER_BEAD
    
    # Initialize NodeMatrix
    node_matrix = NodeMatrix(max_waam_nodes, num_table_nodes, NUMBER_OF_TRACKS, N_ELEMENTS_PER_BEAD)
    
    # --- Table initialization ---
    if TABLE_DISCRETIZATION_MODE == 0:
        # Single node table
        node_matrix.table_idx = node_matrix.add_table_node(
            0, mass=m_table, area=0.0, temperature=AMBIENT_TEMP
        )
        node_matrix.table_indices = [node_matrix.table_idx]
        node_matrix.table_bp_contact_idx = node_matrix.table_idx
        table_info_str = "single node"
    else:
        # Validate dimensions
        node_dx = TABLE_LENGTH / nx
        node_dy = TABLE_WIDTH / ny
        
        if node_dx < BP_LENGTH:
            raise ValueError(
                f"Table node X-dimension ({node_dx*1000:.1f}mm) is smaller than BP_LENGTH ({BP_LENGTH*1000:.1f}mm). "
                f"Reduce TABLE_DISCRETIZATION_MODE or N_TABLE_X, or increase TABLE_LENGTH."
            )
        if node_dy < BP_WIDTH:
            raise ValueError(
                f"Table node Y-dimension ({node_dy*1000:.1f}mm) is smaller than BP_WIDTH ({BP_WIDTH*1000:.1f}mm). "
                f"Reduce TABLE_DISCRETIZATION_MODE or N_TABLE_Y, or increase TABLE_WIDTH."
            )
        
        # Initialize table grid
        node_matrix.initialize_table_grid(
            nx, ny, nz, 
            TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS,
            RHO_TABLE, AMBIENT_TEMP
        )
        
        # Place base plate on top corner node (ix=0, iy=0, iz=nz-1 = top layer)
        node_matrix.table_bp_contact_idx = node_matrix.table_grid[0, 0, nz - 1]
        
        table_info_str = f"{nx}x{ny}x{nz} = {num_table_nodes} nodes (Mode {TABLE_DISCRETIZATION_MODE})"
    
    # Add baseplate node
    node_matrix.add_bp_node(
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
    
    print("Starting simulation...")
    print(f"Layer geometry: {effective_layer_width*1000:.1f}mm x {TRACK_LENGTH*1000:.1f}mm x {LAYER_HEIGHT*1000:.1f}mm")
    print(f"Layer duration: {layer_duration:.1f}s ({NUMBER_OF_TRACKS} tracks at {PROCESS_SPEED*1000:.1f}mm/s)")
    print(f"Total height after {NUMBER_OF_LAYERS} layers: {NUMBER_OF_LAYERS * LAYER_HEIGHT*1000:.1f}mm")
    print(f"Discretization: {N_LAYERS_AS_BEADS} top layers as beads, {N_LAYERS_WITH_ELEMENTS} with elements ({N_ELEMENTS_PER_BEAD}/bead)")
    print(f"Table discretization: {table_info_str}")
    print(f"Max stable DT: {dt_max_stable:.4f} s (using {dt_sim:.4f} s)")
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
                        
                        welding_node_idx = node_matrix.activate_waam_node(
                            layer_idx=i_layer, bead_idx=i_bead, element_idx=i_element,
                            level_type_str='element', mass=element_mass, area=element_area,
                            temperature=MELTING_TEMP
                        )
                        
                        # Simulate welding of this element
                        steps = int(element_duration / dt_sim)
                        for _ in range(steps):
                            node_matrix = update_temperatures_matrix(
                                node_matrix, thermal_model, dt=dt_sim, is_welding=True,
                                arc_power=effective_arc_power, welding_node_idx=welding_node_idx
                            )
                            current_time += dt_sim
                            
                            logging_counter += 1
                            if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                                log_data(current_time, node_matrix, time_log, temp_layers_log, temp_bp_log, temp_table_log)
                else:
                    # Bead-level: add whole bead
                    welding_node_idx = node_matrix.activate_waam_node(
                        layer_idx=i_layer, bead_idx=i_bead, element_idx=-1,
                        level_type_str='bead', mass=bead_mass, area=bead_area,
                        temperature=MELTING_TEMP
                    )
                    
                    # Simulate welding of this bead
                    steps = int(bead_duration / dt_sim)
                    for _ in range(steps):
                        node_matrix = update_temperatures_matrix(
                            node_matrix, thermal_model, dt=dt_sim, is_welding=True,
                            arc_power=effective_arc_power, welding_node_idx=welding_node_idx
                        )
                        current_time += dt_sim
                        
                        logging_counter += 1
                        if logging_counter % LOGGING_EVERY_N_STEPS == 0:
                            log_data(current_time, node_matrix, time_log, temp_layers_log, temp_bp_log, temp_table_log)
        else:
            # Layer-level: add entire layer as single node
            layer_node_idx = node_matrix.activate_waam_node(
                layer_idx=i_layer, bead_idx=-1, element_idx=-1,
                level_type_str='layer', mass=m_layer, area=layer_area,
                temperature=MELTING_TEMP
            )
            
            # Simulate entire layer deposition
            steps = int(layer_duration / dt_sim)
            for _ in range(steps):
                node_matrix = update_temperatures_matrix(
                    node_matrix, thermal_model, dt=dt_sim, is_welding=True,
                    arc_power=effective_arc_power, welding_node_idx=layer_node_idx
                )
                current_time += dt_sim
                
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
            
            if condition_met:
                break
            
            # Safety check for instability
            if t_hottest > 5000.0:
                print(f"Warning: Instability detected (T={t_hottest:.1f}°C). Breaking cooling loop.")
                break

            node_matrix = update_temperatures_matrix(
                node_matrix, thermal_model, dt=dt_sim, is_welding=False
            )
            current_time += dt_sim
            
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
                        node_matrix.consolidate_layer_to_beads(layer_idx)
        
        # Consolidate beads to layer for layers that should now be layer-level
        for layer_idx in range(current_num_layers):
            layers_from_current_top = current_num_layers - layer_idx - 1
            should_be_beads = layers_from_current_top < N_LAYERS_AS_BEADS
            
            if not should_be_beads:
                # This layer should NOT have beads anymore, consolidate to layer
                current_level = node_matrix.get_layer_level_type(layer_idx)
                if current_level == 'bead' or current_level == 'element':
                    node_matrix.consolidate_layer_to_layer(layer_idx)
    
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
    
    # For table temperature, use max if multi-node or single value
    if node_matrix.table_indices and len(node_matrix.table_indices) > 1:
        table_max = max(node_matrix.temperatures[i] for i in node_matrix.table_indices)
        temp_table_log.append(table_max)
    elif node_matrix.table_idx is not None:
        temp_table_log.append(node_matrix.temperatures[node_matrix.table_idx])
    else:
        temp_table_log.append(AMBIENT_TEMP)

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
    plt.plot(t_data, table_data, label='Welding table (Max)', color='grey', linewidth=2)
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