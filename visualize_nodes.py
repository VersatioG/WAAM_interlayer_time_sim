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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default HDF5 state file to visualize.
H5_FILE_NAME = 'simulation_state.h5'

# Maximum number of animation timesteps shown in Plotly.
# Set to 0 or a negative value to show every logged timestep.
MAX_DISPLAYED_TIMESTEPS = 150

# Temperature color scale appearance.
TEMP_SCALE_LENGTH = 0.95
TEMP_SCALE_THICKNESS = 38
TEMP_SCALE_TITLE_FONT_SIZE = 30
TEMP_SCALE_TICK_FONT_SIZE = 23
TEMP_SCALE_COLORSCALE = 'Viridis'  # Can be any Plotly colorscale

# Plot layout and text sizing.
AXIS_TITLE_FONT_SIZE = 24
AXIS_TICK_FONT_SIZE = 18
PLOT_MARGIN_LEFT = 8
PLOT_MARGIN_RIGHT = 8
PLOT_MARGIN_TOP = 70
PLOT_MARGIN_BOTTOM = 50

# UI element placement.
PLAY_BUTTON_X = 0.02
PLAY_BUTTON_Y = 1.07
SLIDER_X = 0.16
SLIDER_LENGTH = 0.80

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

        # Table nodes are stored before the base plate, which is directly before WAAM nodes.
        self.table_end_idx = max(0, self.waam_start_idx - 1)
        self.table_node_indices = np.arange(self.table_end_idx, dtype=np.int32)
        if len(self.table_node_indices) > 0:
            table_bead_codes = self.map_bead[self.table_node_indices]
            table_element_codes = self.map_element[self.table_node_indices]
            table_ix = table_bead_codes // 100
            table_iy = table_bead_codes % 100
            table_iz = table_element_codes
            self.table_grid_shape = (
                int(np.max(table_ix)) + 1,
                int(np.max(table_iy)) + 1,
                int(np.max(table_iz)) + 1,
            )
        else:
            self.table_grid_shape = (1, 1, 1)
            
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
        table_len = self.params.get('TABLE_LENGTH', 2.0)
        table_wid = self.params.get('TABLE_WIDTH', 1.2)
        table_thick = self.params.get('TABLE_THICKNESS', 0.01)
        table_nx, table_ny, table_nz = self.table_grid_shape
        table_dx = table_len / table_nx
        table_dy = table_wid / table_ny
        table_dz = table_thick / table_nz
        table_contact_x = 0.5 * table_dx
        table_contact_y = 0.5 * table_dy
        waam_x_offset = table_contact_x - (tl / 2.0)
        waam_y_offset = table_contact_y - layer_center_y
        
        node_names = []

        for i, idx in enumerate(active_indices):
            l_val = l_types[idx]
            
            # --- NON-WAAM NODES (Table/BP) ---
            if idx < self.waam_start_idx:
                if idx == self.waam_start_idx - 1: # BP
                    centers[i] = [table_contact_x, table_contact_y, bp_thick/2]
                    sizes[i] = [bp_len, bp_wid, bp_thick]
                    node_names.append(f"Base Plate (Node {idx})")
                else: # Table (Simplified)
                    ix = int(self.map_bead[idx]) // 100
                    iy = int(self.map_bead[idx]) % 100
                    iz = int(self.map_element[idx])

                    centers[i] = [
                        (ix + 0.5) * table_dx,
                        (iy + 0.5) * table_dy,
                        -((table_nz - iz - 0.5) * table_dz),
                    ]
                    sizes[i] = [table_dx, table_dy, table_dz]
                    node_names.append(f"Table Node {idx}")
                continue
            
            # --- WAAM NODES ---
            L = self.map_layer[idx]
            B = self.map_bead[idx]
            E = self.map_element[idx]
            
            cz = bp_thick + L * lh + lh/2
            
            if l_val == 1: # TYPE_LAYER
                centers[i] = [waam_x_offset + tl/2, waam_y_offset + layer_center_y, cz]
                sizes[i] = [tl, layer_width, lh]
                node_names.append(f"Layer {L} (Node {idx})")
                
            elif l_val == 2: # TYPE_BEAD
                centers[i] = [waam_x_offset + tl/2, waam_y_offset + B * pitch, cz]
                sizes[i] = [tl, tw, lh]
                node_names.append(f"L{L} Bead {B} (Node {idx})")
                
            elif l_val == 3: # TYPE_ELEMENT
                centers[i] = [waam_x_offset + E * elem_len + elem_len/2, waam_y_offset + B * pitch, cz]
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
    
    scat = ax.scatter(
        centers[:,0], centers[:,1], centers[:,2],
        c=temps, cmap=TEMP_SCALE_COLORSCALE, marker='s', s=60,
        edgecolors='black', linewidths=0.35
    )
    cb = plt.colorbar(scat, fraction=0.07, pad=0.04)
    cb.set_label('Temperature [°C]', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
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
        scat.set_clim(vmin=t.min(), vmax=t.max())
        
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


def generate_edge_data(centers, sizes):
    """Generate line segments for cube edges."""
    edge_x = []
    edge_y = []
    edge_z = []

    # Only draw the faces that are visible from the default camera direction.
    face_edges = [
        [(1, 2), (2, 6), (6, 5), (5, 1)],
        [(2, 3), (3, 7), (7, 6), (6, 2)],
        [(4, 5), (5, 6), (6, 7), (7, 4)],
    ]

    for center, size in zip(centers, sizes):
        x0, y0, z0 = center - size / 2.0
        x1, y1, z1 = center + size / 2.0

        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])

        for edges in face_edges:
            for a, b in edges:
                edge_x.extend([corners[a, 0], corners[b, 0], None])
                edge_y.extend([corners[a, 1], corners[b, 1], None])
                edge_z.extend([corners[a, 2], corners[b, 2], None])

    return edge_x, edge_y, edge_z

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
    edge_x, edge_y, edge_z = generate_edge_data(centers, sizes)
    
    # Base Trace
    trace = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=intensity,
        text=np.repeat([f"{Name}<br>Temp: {T:.1f} °C" for Name, T in zip(node_names, temps)], 8),
        hoverinfo='text',
        colorscale=TEMP_SCALE_COLORSCALE,
        colorbar=dict(
            title=dict(text='Temp [°C]', font=dict(size=TEMP_SCALE_TITLE_FONT_SIZE)),
            tickfont=dict(size=TEMP_SCALE_TICK_FONT_SIZE),
            len=TEMP_SCALE_LENGTH,
            thickness=TEMP_SCALE_THICKNESS,
            x=1.01,
            xanchor='left',
        ),
        name='Nodes',
        showscale=True,
        flatshading=True,
        opacity=0.95
    )

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(20,20,20,0.7)', width=2),
        hoverinfo='skip',
        showlegend=False,
        name='Edges'
    )
    
    # Layout
    layout = go.Layout(
        title=f"Time: {time:.2f} s",
        scene=dict(
            xaxis=dict(
                title=dict(text='X [m]', font=dict(size=AXIS_TITLE_FONT_SIZE)),
                tickfont=dict(size=AXIS_TICK_FONT_SIZE)
            ),
            yaxis=dict(
                title=dict(text='Y [m]', font=dict(size=AXIS_TITLE_FONT_SIZE)),
                tickfont=dict(size=AXIS_TICK_FONT_SIZE)
            ),
            zaxis=dict(
                title=dict(text='Z [m]', font=dict(size=AXIS_TITLE_FONT_SIZE)),
                tickfont=dict(size=AXIS_TICK_FONT_SIZE)
            ),
            aspectmode='data',  # Crucial for correct proportions
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.9))
        ),
        updatemenus=[dict(
            type="buttons",
            x=PLAY_BUTTON_X,
            y=PLAY_BUTTON_Y,
            xanchor='left',
            yanchor='bottom',
            pad=dict(t=0, r=0),
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=100, redraw=True), 
                                           fromcurrent=True)])]
        )],
        margin=dict(l=PLOT_MARGIN_LEFT, r=PLOT_MARGIN_RIGHT, t=PLOT_MARGIN_TOP, b=PLOT_MARGIN_BOTTOM)
    )
    
    # Frames
    frames = []
    # Use actual logged steps. If the file was logged with LOGGING_FREQUENCY=0.5, then the steps are already 0.5s apart.
    # If using stride=1, we show every logged step.
    # Current stride calculates "50 frames total", which might be too coarse or fine.
    # Let's set stride=1 if user wants full resolution, or auto-downsample only if massive.
    
    if MAX_DISPLAYED_TIMESTEPS and MAX_DISPLAYED_TIMESTEPS > 0:
        stride = max(1, loader.num_steps // MAX_DISPLAYED_TIMESTEPS)
    else:
        stride = 1
    
    print(f"Generating animation frames (Stride={stride})...")
    for step_idx in range(0, loader.num_steps, stride):
        c, s, t, ty, tm, idxs, names = loader.get_step_data(step_idx)
        mx, my, mz, mi, mj, mk, mint = generate_mesh_data(c, s, t)
        ex, ey, ez = generate_edge_data(c, s)
        
        # Build hover text for this frame
        # We need to repeat the text 8 times (once per vertex)
        frame_text = np.repeat([f"{Name}<br>Temp: {T:.1f} °C" for Name, T in zip(names, t)], 8)

        frames.append(go.Frame(
            data=[
                go.Mesh3d(
                    x=mx, y=my, z=mz,
                    i=mi, j=mj, k=mk,
                    intensity=mint,
                    text=frame_text,
                    hoverinfo='text',
                    colorscale=TEMP_SCALE_COLORSCALE,
                    flatshading=True,
                    opacity=1.0
                ),
                go.Scatter3d(
                    x=ex, y=ey, z=ez,
                    mode='lines',
                    line=dict(color='rgba(20,20,20,0.8)', width=2),
                    hoverinfo='skip',
                    showlegend=False
                )
            ],
            name=str(step_idx),
            layout=go.Layout(title=f"Time: {tm:.2f} s")
        ))

    fig = go.Figure(data=[trace, edge_trace], layout=layout, frames=frames)
    
    # Slider with time labels
    sliders = [dict(
        x=SLIDER_X,
        len=SLIDER_LENGTH,
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
    parser.add_argument('file', nargs='?', default=H5_FILE_NAME, help='HDF5 State File')
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
