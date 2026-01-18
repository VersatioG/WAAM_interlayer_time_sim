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
        
        # Read active indices and temps
        temps = self.temps[step_idx]
        active = self.active_mask[step_idx].astype(bool)
        l_types = self.level_type[step_idx] # 0=Inactive, 1=Layer, 2=Bead, 3=Element
        
        active_indices = np.where(active)[0]
        
        # We will build numpy arrays for results
        centers = np.zeros((len(active_indices), 3))
        sizes = np.zeros((len(active_indices), 3))
        
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
        
        node_names = []

        for i, idx in enumerate(active_indices):
            l_val = l_types[idx]
            
            # --- NON-WAAM NODES (Table/BP) ---
            if idx < self.waam_start_idx:
                if idx == self.waam_start_idx - 1: # BP
                    centers[i] = [tl/2, layer_center_y, -bp_thick/2]
                    sizes[i] = [bp_len, bp_wid, bp_thick]
                    node_names.append(f"Base Plate (Node {idx})")
                else: # Table (Simplified)
                    # Distribute table nodes slightly to verify they are not all just one
                    # Simple grid attempt? No, just stack them slightly offset or same place
                    centers[i] = [tl/2, layer_center_y, -0.02 - bp_thick - (idx * 0.001)] 
                    sizes[i] = [0.5, 0.5, 0.02]
                    node_names.append(f"Table Node {idx}")
                continue
            
            # --- WAAM NODES ---
            L = self.map_layer[idx]
            B = self.map_bead[idx]
            E = self.map_element[idx]
            
            cz = L * lh + lh/2
            
            if l_val == 1: # TYPE_LAYER
                centers[i] = [tl/2, layer_center_y, cz]
                sizes[i] = [tl, layer_width, lh]
                node_names.append(f"Layer {L} (Node {idx})")
                
            elif l_val == 2: # TYPE_BEAD
                centers[i] = [tl/2, B * pitch, cz]
                sizes[i] = [tl, tw, lh]
                node_names.append(f"L{L} Bead {B} (Node {idx})")
                
            elif l_val == 3: # TYPE_ELEMENT
                centers[i] = [E * elem_len + elem_len/2, B * pitch, cz]
                sizes[i] = [elem_len, tw, lh]
                node_names.append(f"L{L} B{B} E{E} (Node {idx})")
            else:
                node_names.append(f"Unknown {idx}")
            
        return centers, sizes, temps[active_indices], l_types[active_indices], self.time[step_idx], active_indices, node_names

    def get_table_temp_range(self):
        """Get min/max temperature of table nodes across all time."""
        if self.waam_start_idx > 0:
            table_temps = self.temps[:, :self.waam_start_idx]
            return np.min(table_temps), np.max(table_temps), np.mean(table_temps)
        return None, None, None

def visualize_matplotlib(loader):
    """Simple Matplotlib Visualization."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)
    
    # Load Initial
    centers, sizes, temps, _, time, _, _ = loader.get_step_data(0)
    
    if len(centers) == 0:
        print("No active nodes in step 0.")
        return
        
    print(f"Plotting {len(centers)} nodes...")

    # Bar3D for blocks (Slow but accurate size)
    # We use a trick: only plot markers but scale them? No, user requested "original size blocks".
    # Bar3d is really slow for animations. We will stick to scatter for Matplotlib speed, 
    # but use bar3d for the initial static frame if requested.
    # Actually, let's use scatter but scaled to roughly reflect size?
    # No, let's try to make it right.
    # For performance, we stick to Scatter in Matplotlib but show a warning that Plotly is better.
    
    scat = ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=temps, cmap='hot', marker='s', s=50)
    cb = plt.colorbar(scat)
    cb.set_label('Temperature [°C]')
    
    title = ax.set_title(f"Time: {time:.2f} s")
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # Scale axes
    all_limits = np.concatenate([centers - sizes/2, centers + sizes/2])
    if len(all_limits) > 0:
        ax.set_xlim(all_limits[:,0].min(), all_limits[:,0].max())
        ax.set_ylim(all_limits[:,1].min(), all_limits[:,1].max())
        ax.set_zlim(all_limits[:,2].min(), all_limits[:,2].max())

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    # Slider maps index to time string
    slider = Slider(ax_slider, 'Time', 0, loader.num_steps-1, valinit=0, valstep=1)
    
    def format_coord(val):
        idx = int(val)
        if 0 <= idx < loader.num_steps:
            return f"{loader.time[idx]:.1f}s"
        return ""
    
    # Hack to show time on slider
    slider.valtext.set_text(format_coord(0))

    def update(val):
        step = int(slider.val)
        c, s, t, _, time_val, _, _ = loader.get_step_data(step)
        
        if len(c) == 0:
            return
            
        scat._offsets3d = (c[:,0], c[:,1], c[:,2])
        scat.set_array(t)
        scat.set_clim(vpn=t.min(), vmax=t.max())
        
        title.set_text(f"Time: {time_val:.2f} s")
        slider.valtext.set_text(f"{time_val:.1f} s")
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    plt.show()

def generate_mesh_data(centers, sizes, temps):
    """Generate vertices and faces for direct mesh visualization of cubes."""
    N = len(centers)
    
    # Relative vertices of a unit cube (centered at 0)
    # 8 vertices
    dx = sizes[:, 0] / 2.0
    dy = sizes[:, 1] / 2.0
    dz = sizes[:, 2] / 2.0
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    cz = centers[:, 2]
    
    # Arrays to hold all vertices (8 * N)
    x = np.empty(N * 8)
    y = np.empty(N * 8)
    z = np.empty(N * 8)
    
    # 0: -x -y -z
    x[0::8] = cx - dx; y[0::8] = cy - dy; z[0::8] = cz - dz
    # 1: +x -y -z
    x[1::8] = cx + dx; y[1::8] = cy - dy; z[1::8] = cz - dz
    # 2: +x +y -z
    x[2::8] = cx + dx; y[2::8] = cy + dy; z[2::8] = cz - dz
    # 3: -x +y -z
    x[3::8] = cx - dx; y[3::8] = cy + dy; z[3::8] = cz - dz
    # 4: -x -y +z
    x[4::8] = cx - dx; y[4::8] = cy - dy; z[4::8] = cz + dz
    # 5: +x -y +z
    x[5::8] = cx + dx; y[5::8] = cy - dy; z[5::8] = cz + dz
    # 6: +x +y +z
    x[6::8] = cx + dx; y[6::8] = cy + dy; z[6::8] = cz + dz
    # 7: -x +y +z
    x[7::8] = cx - dx; y[7::8] = cy + dy; z[7::8] = cz + dz
    
    # Indices for triangles (12 triangles * 3 vertices = 36 indices per cube)
    # But for Mesh3d we need i, j, k arrays (each length 12 * N)
    
    # Template for one cube (12 triangles)
    # Faces: Bottom(0,1,2,3), Top(4,5,6,7), Front(0,1,5,4), Right(1,2,6,5), Back(2,3,7,6), Left(3,0,4,7)
    # Triangles: (0,1,2), (0,2,3) ...
    
    base_i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    base_j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    base_k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    # Wait, simple decomposition:
    # 6 faces * 2 tris = 12 tris
    # Bottom: 0-2-1, 0-3-2  (Correct normal pointing down? Let's generic)
    # Order: 0,1,2; 0,2,3 (Bottom z-)
    #        4,5,6; 4,6,7 (Top z+)
    #        0,1,5; 0,5,4 (Front y-)
    #        1,2,6; 1,6,5 (Right x+)
    #        2,3,7; 2,7,6 (Back y+)
    #        3,0,4; 3,4,7 (Left x-)
    
    bi = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    bj = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    bk = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    
    # Vectorize indices
    i_idxs = np.array(bi)
    j_idxs = np.array(bj)
    k_idxs = np.array(bk)
    
    # Replicate for N cubes with offset
    # Shape (N, 12) -> flatten
    offsets = np.arange(N) * 8
    
    II = (i_idxs[np.newaxis, :] + offsets[:, np.newaxis]).flatten()
    JJ = (j_idxs[np.newaxis, :] + offsets[:, np.newaxis]).flatten()
    KK = (k_idxs[np.newaxis, :] + offsets[:, np.newaxis]).flatten()
    
    # Intensity (Temperature) - replicated 8 times per cube
    intensities = np.repeat(temps, 8)
    
    return x, y, z, II, JJ, KK, intensities

def visualize_plotly(loader):
    """Interactive Plotly Visualization with Scaled Blocks."""
    print("Preparing Plotly visualization...")
    
    t_min, t_max, t_mean = loader.get_table_temp_range()
    if t_min is not None:
        print(f"Table Temp Range: [{t_min:.2f}, {t_max:.2f}] °C (Mean: {t_mean:.2f})")
    
    # Initial data
    centers, sizes, temps, types, time, active_indices, node_names = loader.get_step_data(0)
    
    # Generate Mesh
    x, y, z, i, j, k, intensity = generate_mesh_data(centers, sizes, temps)
    
    # Base Trace
    trace = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=intensity,
        text=np.repeat([f"{Name}<br>Temp: {T:.1f} °C" for Name, T in zip(node_names, temps)], 8),
        hoverinfo='text',
        colorscale='Hot',
        colorbar=dict(title='Temp [°C]'),
        name='Nodes',
        showscale=True,
        flatshading=True 
    )
    
    # Layout
    layout = go.Layout(
        title=f"Time: {time:.2f} s",
        scene=dict(
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            zaxis_title='Z [m]',
            aspectmode='data'  # Crucial for correct proportions
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=100, redraw=True), 
                                           fromcurrent=True)])]
        )]
    )
    
    # Frames
    frames = []
    # Use actual logged steps. If the file was logged with LOGGING_FREQUENCY=0.5, then the steps are already 0.5s apart.
    # If using stride=1, we show every logged step.
    # Current stride calculates "50 frames total", which might be too coarse or fine.
    # Let's set stride=1 if user wants full resolution, or auto-downsample only if massive.
    
    # Using stride=1 (every logged step)
    stride = 1 
    if loader.num_steps > 300:
        stride = loader.num_steps // 100 # Limit to ~100 frames for performance if huge
    
    print(f"Generating animation frames (Stride={stride})...")
    for step_idx in range(0, loader.num_steps, stride):
        c, s, t, ty, tm, idxs, names = loader.get_step_data(step_idx)
        mx, my, mz, mi, mj, mk, mint = generate_mesh_data(c, s, t)
        
        # Build hover text for this frame
        # We need to repeat the text 8 times (once per vertex)
        frame_text = np.repeat([f"{Name}<br>Temp: {T:.1f} °C" for Name, T in zip(names, t)], 8)

        frames.append(go.Frame(
            data=[go.Mesh3d(
                x=mx, y=my, z=mz,
                i=mi, j=mj, k=mk,
                intensity=mint,
                text=frame_text,
                hoverinfo='text'
            )],
            name=str(step_idx),
            layout=go.Layout(title=f"Time: {tm:.2f} s")
        ))

    fig = go.Figure(data=[trace], layout=layout, frames=frames)
    
    # Slider with time labels
    sliders = [dict(
        steps=[dict(
            method= 'animate',
            args= [[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=f"{loader.time[k]:.1f}s"
        ) for k in range(0, loader.num_steps, stride)],
        currentvalue=dict(prefix='Time: '),
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
