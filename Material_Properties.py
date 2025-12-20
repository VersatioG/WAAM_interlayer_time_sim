import numpy as np
# from scipy.interpolate import interp1d # Removed for performance

class Material:
    """
    Base class for material properties.
    """
    def __init__(self, name):
        self.name = name
        # self._interp_func = None # Deprecated

    def get_cp(self, T_c):
        """
        Returns specific heat capacity [J/kgK] at temperature T_c [°C].
        """
        raise NotImplementedError("Material specific heat capacity not implemented.")

class S235JR(Material):
    """
    Thermodynamic modeling of specific heat capacity for structural steel S235JR
    from 0°C to 2500°C.
    
    Based on:
    - Eurocode 3 (EN 1993-1-2) for solid phase (0-1200°C)
    - Effective Heat Capacity Method for melting range
    - NIST/Shomate data for liquid phase
    """
    
    def __init__(self):
        super().__init__("S235JR")
        # Physical Constants
        self.T_solidus = 1517.0  # °C
        self.T_liquidus = 1535.0 # °C
        self.H_m = 113500.0      # J/kg (Latent heat of fusion)
        self.cp_liquid = 824.0   # J/kgK (Constant approximation for liquid phase)
        
        # Generate lookup table for efficiency
        self._create_lookup_table()
        
    def _cp_eurocode(self, T_c):
        """
        Eurocode 3 Part 1-2 specific heat capacity for carbon steel.
        T_c: Temperature in Celsius (scalar or array)
        """
        T_c = np.atleast_1d(T_c)
        cp = np.zeros_like(T_c, dtype=np.float64)
        
        # Range 1: 20 <= T < 600
        # We extend this down to 0 for simulation purposes
        mask1 = (T_c < 600.0)
        t = T_c[mask1]
        # Formula: 425 + 7.73e-1*t - 1.69e-3*t^2 + 2.22e-6*t^3
        cp[mask1] = 425.0 + 0.773 * t - 1.69e-3 * t**2 + 2.22e-6 * t**3
        
        # Range 2: 600 <= T < 735
        mask2 = (T_c >= 600.0) & (T_c < 735.0)
        t = T_c[mask2]
        # Formula: 666 + 13002 / (738 - t)
        cp[mask2] = 666.0 + 13002.0 / (738.0 - t)
        
        # Range 3: 735 <= T < 900
        mask3 = (T_c >= 735.0) & (T_c < 900.0)
        t = T_c[mask3]
        # Formula: 545 + 17820 / (t - 731)
        cp[mask3] = 545.0 + 17820.0 / (t - 731.0)
        
        # Range 4: 900 <= T <= 1200
        mask4 = (T_c >= 900.0) & (T_c <= 1200.0)
        cp[mask4] = 650.0
        
        return cp

    def _create_lookup_table(self):
        """
        Creates a dense lookup table for efficient interpolation.
        Range: 0°C to 3000°C
        Resolution: 0.5 K
        """
        T_min = 0.0
        T_max = 3000.0
        step = 0.5
        self.T_table = np.arange(T_min, T_max + step, step)
        
        # 1. Base Sensible Heat Capacity
        cp_sensible = np.zeros_like(self.T_table)
        
        # Solid Phase (Eurocode 3) up to 1200°C
        mask_solid = self.T_table <= 1200.0
        cp_sensible[mask_solid] = self._cp_eurocode(self.T_table[mask_solid])
        
        # Transition 1200°C to Solidus (1517°C)
        # Constant 650 J/kgK as per recommendation ("Fortführung des Plateaus")
        mask_trans = (self.T_table > 1200.0) & (self.T_table <= self.T_solidus)
        cp_sensible[mask_trans] = 650.0
        
        # Liquid Phase (> Liquidus)
        mask_liquid = self.T_table >= self.T_liquidus
        cp_sensible[mask_liquid] = self.cp_liquid
        
        # Melting Range Sensible (Linear interpolation between 650 and 824)
        mask_melt = (self.T_table > self.T_solidus) & (self.T_table < self.T_liquidus)
        if np.any(mask_melt):
            t_melt = self.T_table[mask_melt]
            # Linear interpolation
            frac = (t_melt - self.T_solidus) / (self.T_liquidus - self.T_solidus)
            cp_sensible[mask_melt] = 650.0 + (self.cp_liquid - 650.0) * frac
            
        # 2. Latent Heat Contribution (Gaussian Smoothing)
        # Center of the melting interval
        T_center = (self.T_solidus + self.T_liquidus) / 2.0
        # Width of the interval
        dT = self.T_liquidus - self.T_solidus
        # Sigma for Gaussian: 
        # We want the area under the curve to be H_m.
        # We want the curve to be mostly contained within [T_sol, T_liq].
        # 4 sigma = dT => sigma = dT / 4
        sigma = dT / 4.0
        
        # Gaussian function: f(T) = A * exp( -0.5 * ((T - mu)/sigma)^2 )
        # Integral = A * sigma * sqrt(2*pi) = H_m
        # A = H_m / (sigma * sqrt(2*pi))
        A = self.H_m / (sigma * np.sqrt(2 * np.pi))
        
        cp_latent = A * np.exp(-0.5 * ((self.T_table - T_center) / sigma)**2)
        
        # Total Effective Heat Capacity
        self.cp_table = cp_sensible + cp_latent
        
        # Optimization: Use numpy arrays for fast interpolation instead of scipy object
        # self._interp_func = interp1d(self.T_table, self.cp_table, kind='linear', fill_value="extrapolate")

    def get_cp(self, T_c):
        """
        Returns specific heat capacity [J/kgK] at temperature T_c [°C].
        Uses numpy.interp for faster execution than scipy.interpolate.interp1d.
        """
        return np.interp(T_c, self.T_table, self.cp_table)

# Material Database
_materials = {
    "S235JR": S235JR()
}

def get_material(name):
    """
    Factory function to get a material instance by name.
    """
    if name in _materials:
        return _materials[name]
    else:
        raise ValueError(f"Material '{name}' not found in database. Available: {list(_materials.keys())}")

def get_cp_s235jr(T_c):
    """
    Legacy wrapper for backward compatibility if needed, 
    or direct access to S235JR.
    """
    return _materials["S235JR"].get_cp(T_c)
