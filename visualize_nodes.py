"""
3D Node Temperature Visualization for WAAM Thermal Simulation
Adapted for HDF5 logging format.
"""

import numpy as np
import h5py
import argparse
import sys
import json
import six

# Try to import plotly for interactive visualization
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Fallback to matplotlib
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.widgets import Slider
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Constants (Defaults if missing in file)
DEFAULTS = {
    'TRACK_WIDTH': 0.0063,
    'TRACK_OVERLAP': 0.738,
    'TRACK_LENGTH': 0.15,
    'LAYER_HEIGHT': 0.002,
    'NUMBER_OF_TRACKS': 4,
    'BP_LENGTH': 0.15,
    'BP_WIDTH': 0.15,
    'BP_THICKNESS': 0.01,
}

class SimulationDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = h5py.File(filepath, 'r')
        
        # Load Parameters
        try:
            # Handle string decoding if necessary
            param_str = self.file.attrs['parameters']
            if isinstance(param_str, bytes):
                param_str = param_str.decode('utf-8')
            self.params = json.loads(param_str)
        except Exception as e:
            print(f"Warning: Could not load parameters from HDF5 attributes: {e}")
            self.params = {}
            
        # Merge defaults
        for k, v in DEFAULTS.items():
            if k not in self.params:
                self.params[k] = v

        # Load Datasets
        self.time = self.file['time'][:]
        self.temps = self.file['temperatures']
        self.active_mask = self.file['active_mask']
        self.level_type = self.file['level_type']
        
        # Static Mappings
        try:
            self.map_layer = self.file['node_map_layer'][:]
            self.map_bead = self.file['node_map_bead'][:]
            self.map_element = self.file['node_map_element'][:]
        except KeyError:
            print("Error: Node mapping datasets not found. Please run the simulation again with the updated code.")
            sys.exit(1)
            
        self.num_steps = len(self.time)
        self.total_nodes = self.temps.shape[1]
        
        # Determine Base Plate and Table indices
        # WAAM indices are where map_layer >= 0
        waam_indices = np.where(self.map_layer >= 0)[0]
        if len(waam_indices) > 0:
            self.waam_start_idx = waam_indices[0]
        else:
            self.waam_start_idx = self.total_nodes 
            
        print(f"File loaded: {self.num_steps} steps, {self.total_nodes} nodes.")

    def get_step_data(self, step_idx):
        """Reconstruct geometry and temperatures for a given step."""
        
        # Read arrays for this step
        temps = self.temps[step_idx]
        active = self.active_mask[step_idx].astype(bool)
        l_types = self.level_type[step_idx] # 0=Inactive, 1=Layer, 2=Bead, 3=Element
        
        # Filter active indices
        active_indices = np.where(active)[0]
        
        # We will build numpy arrays for results
        centers = np.zeros((len(active_indices), 3))
        sizes = np.zeros((len(active_indices), 3))
        result_temps = np.zeros(len(active_indices))
        types = [] # String list
        
        # Cache params
        tw = self.params.get('TRACK_WIDTH', DEFAULTS['TRACK_WIDTH'])
        tl = self.params.get('TRACK_LENGTH', DEFAULTS['TRACK_LENGTH'])
        lh = self.params.get('LAYER_HEIGHT', DEFAULTS['LAYER_HEIGHT'])
        overlap = self.params.get('TRACK_OVERLAP', DEFAULTS['TRACK_OVERLAP'])
        n_tracks = self.params.get('NUMBER_OF_TRACKS', DEFAULTS['NUMBER_OF_TRACKS'])
        
        # Pitch between beads
        pitch = tw * overlap
        
        # Calculate Effective Layer Width
        layer_width = (n_tracks - 1) * pitch + tw
        layer_center_y = ((n_tracks - 1) * pitch) / 2.0
        
        # Element Length approximation
        n_elements = self.params.get('N_ELEMENTS_PER_BEAD', 20) 
        elem_len = tl / n_elements
        
        bp_thick = self.params.get('BP_THICKNESS', DEFAULTS['BP_THICKNESS'])
        bp_len = self.params.get('BP_LENGTH', DEFAULTS['BP_LENGTH'])
        bp_wid = self.params.get('BP_WIDTH', DEFAULTS['BP_WIDTH'])
        
        # Vectorized approach or Loop? Loop is safer for mixed types.
        # But we can split by type for speed if needed.
        # Let's do simple loop for clarity first.
        
        for i, idx in enumerate(active_indices):
            t_val = temps[idx]
            l_val = l_types[idx]
            
            result_temps[i] = t_val
            
            # --- NON-WAAM NODES (Table/BP) ---
            if idx < self.waam_start_idx:
                if idx == self.waam_start_idx - 1: # BP
                    centers[i] = [tl/2, layer_center_y, -bp_thick/2]
                    sizes[i] = [bp_len, bp_wid, bp_thick]
                    types.append('BP')
                else: # Table (Simplified)
                    centers[i] = [tl/2, layer_center_y, -0.02 - bp_thick]
                    sizes[i] = [0.5, 0.5, 0.02]
                    types.append('Table')
                continue
            
            # --- WAAM NODES ---
            L = self.map_layer[idx]
            B = self.map_bead[idx]
            E = self.map_element[idx]
            
            cz = L * lh + lh/2
            
            if l_val == 1: # TYPE_LAYER
                centers[i] = [tl/2, layer_center_y, cz]
                sizes[i] = [tl, layer_width, lh]
                types.append('Layer')
                
            elif l_val == 2: # TYPE_BEAD
                centers[i] = [tl/2, B * pitch, cz]
                sizes[i] = [tl, tw, lh]
                types.append('Bead')
                
            elif l_val == 3: # TYPE_ELEMENT
                centers[i] = [E * elem_len + elem_len/2, B * pitch, cz]
                sizes[i] = [elem_len, tw, lh]
                types.append('Element')
            else:
                types.append('Unknown')
            
        return centers, sizes, result_temps, types, self.time[step_idx]

def visualize_matplotlib(loader):
    """Simple Matplotlib Visualization."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)
    
    # Load Initial
    centers, _, temps, _, time = loader.get_step_data(0)
    
    if len(centers) == 0:
        print("No active nodes in step 0.")
        return

    # Scatter plot
    scat = ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=temps, cmap='hot', marker='s', s=20)
    cb = plt.colorbar(scat)
    cb.set_label('Temperature [°C]')
    
    title = ax.set_title(f"Time: {time:.2f} s")
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # Initial limits
    if len(centers) > 0:
        ax.set_xlim(centers[:,0].min()-0.02, centers[:,0].max()+0.02)
        ax.set_ylim(centers[:,1].min()-0.02, centers[:,1].max()+0.02)
        ax.set_zlim(centers[:,2].min()-0.01, centers[:,2].max()+0.01)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Step', 0, loader.num_steps-1, valinit=0, valstep=1)
    
    def update(val):
        step = int(slider.val)
        c, _, t, _, time_val = loader.get_step_data(step)
        
        if len(c) == 0:
            return
            
        # Update positions and colors
        scat._offsets3d = (c[:,0], c[:,1], c[:,2])
        scat.set_array(t)
        scat.set_clim(vpn=t.min(), vmax=t.max())
        
        title.set_text(f"Time: {time_val:.2f} s")
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    plt.show()

def visualize_plotly(loader):
    """Interactive Plotly Visualization."""
    print("Preparing Plotly visualization...")
    
    # Initial data
    centers, sizes, temps, types, time = loader.get_step_data(0)
    
    # Base Trace
    trace = go.Scatter3d(
        x=centers[:,0], y=centers[:,1], z=centers[:,2],
        mode='markers',
        marker=dict(
            size=5,
            color=temps,
            colorscale='Hot',
            colorbar=dict(title='Temp [°C]'),
            opacity=1.0,
            symbol='square'
        ),
        text=[f"Type: {ty}<br>T: {t:.1f}" for ty, t in zip(types, temps)],
        name='Nodes'
    )
    
    # Layout
    layout = go.Layout(
        title=f"Time: {time:.2f} s",
        scene=dict(
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            zaxis_title='Z [m]',
            aspectmode='data'
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=100, redraw=True), 
                                           fromcurrent=True)])]
        )]
    )
    
    # Frames (Downsample if too many)
    frames = []
    stride = max(1, loader.num_steps // 100) # Aim for max 100 frames
    
    print(f"Generating animation frames (Stride={stride})...")
    for i in range(0, loader.num_steps, stride):
        c, _, t, ty, tm = loader.get_step_data(i)
        
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=c[:,0], y=c[:,1], z=c[:,2],
                marker=dict(color=t),
                text=[f"Type: {typ}<br>T: {val:.1f}" for typ, val in zip(ty, t)]
            )],
            name=str(i),
            layout=go.Layout(title=f"Time: {tm:.2f} s")
        ))

    fig = go.Figure(data=[trace], layout=layout, frames=frames)
    
    # Slider
    sliders = [dict(
        steps=[dict(
            method= 'animate',
            args= [[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=str(k)
        ) for k in range(0, loader.num_steps, stride)],
        currentvalue=dict(prefix='Step: '),
        pad=dict(t=50)
    )]
    fig.update_layout(sliders=sliders)
    
    print("Showing plot...")
    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize WAAM Thermal Simulation HDF5")
    parser.add_argument('file', nargs='?', default='simulation_state.h5', help='HDF5 State File')
    parser.add_argument('--backend', choices=['matplotlib', 'plotly'], default='plotly' if HAS_PLOTLY else 'matplotlib')
    
    args = parser.parse_args()
    
    if not args.file:
        print("Please provide a file.")
        return

    try:
        loader = SimulationDataLoader(args.file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if args.backend == 'plotly' and HAS_PLOTLY:
        visualize_plotly(loader)
    else:
        visualize_matplotlib(loader)

if __name__ == "__main__":
    main()
