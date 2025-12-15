"""
3D Node Temperature Visualization for WAAM Thermal Simulation

This script visualizes the temperature data logged by log_node_temperatures.py.
Each node is displayed as a 3D block with color representing temperature.
An interactive slider allows navigation through timesteps.

Usage:
    python visualize_nodes.py [--file FILE] [--colormap CMAP] [--temp-min MIN] [--temp-max MAX]

Example:
    python visualize_nodes.py --file node_temperatures.h5 --colormap hot
    
Controls:
    - Use slider to navigate timesteps
    - Click Play to animate
    - Mouse: rotate/zoom the 3D view
"""

import numpy as np
import h5py
import argparse
import sys

# Try to import plotly for interactive visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Fallback to matplotlib
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.widgets import Slider, Button
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_hdf5_data(filepath):
    """
    Load temperature data from HDF5 file.
    
    Returns:
        dict with keys:
            - timesteps: array of time values
            - snapshots: list of dicts with temperatures, positions, dimensions, types
            - metadata: dict with simulation parameters
    """
    data = {
        'timesteps': None,
        'snapshots': [],
        'metadata': {}
    }
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        for key in f.attrs.keys():
            data['metadata'][key] = f.attrs[key]
        
        # Load timesteps
        data['timesteps'] = f['timesteps'][:]
        
        # Load wait times if available
        if 'wait_times' in f:
            data['wait_times'] = f['wait_times'][:]
        
        # Load snapshots
        snapshots_grp = f['snapshots']
        num_snapshots = data['metadata'].get('num_snapshots', len(snapshots_grp.keys()))
        
        type_map = {0: 'table', 1: 'baseplate', 2: 'layer', 3: 'bead', 4: 'element', 5: 'unknown'}
        
        for i in range(num_snapshots):
            snap_key = f'{i:06d}'
            if snap_key in snapshots_grp:
                snap = snapshots_grp[snap_key]
                snapshot = {
                    'temperatures': snap['temperatures'][:],
                    'positions': snap['positions'][:],
                    'dimensions': snap['dimensions'][:],
                    'types': [type_map.get(t, 'unknown') for t in snap['types'][:]]
                }
                data['snapshots'].append(snapshot)
    
    return data


def create_cube_mesh(center, dims, scale=1.0):
    """
    Create vertices and faces for a cube mesh.
    
    Args:
        center: (x, y, z) center position
        dims: (dx, dy, dz) dimensions
        scale: scaling factor for visualization
    
    Returns:
        vertices: 8x3 array of vertex positions
        faces: list of 6 faces, each with 4 vertex indices
    """
    cx, cy, cz = center
    dx, dy, dz = [d * scale for d in dims]
    
    # 8 vertices of the cube
    vertices = np.array([
        [cx - dx/2, cy - dy/2, cz - dz/2],
        [cx + dx/2, cy - dy/2, cz - dz/2],
        [cx + dx/2, cy + dy/2, cz - dz/2],
        [cx - dx/2, cy + dy/2, cz - dz/2],
        [cx - dx/2, cy - dy/2, cz + dz/2],
        [cx + dx/2, cy - dy/2, cz + dz/2],
        [cx + dx/2, cy + dy/2, cz + dz/2],
        [cx - dx/2, cy + dy/2, cz + dz/2],
    ])
    
    # 6 faces (each defined by 4 vertices)
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]
    
    return vertices, faces


def visualize_with_plotly(data, colormap='hot', temp_min=None, temp_max=None, max_frames=100):
    """
    Create interactive 3D visualization using Plotly.
    
    Args:
        max_frames: Maximum number of frames to display (subsamples if more)
    """
    print("Creating Plotly visualization...")
    
    # Subsample if too many snapshots
    n_snapshots = len(data['snapshots'])
    if n_snapshots > max_frames:
        step = n_snapshots // max_frames
        indices = list(range(0, n_snapshots, step))[:max_frames]
        print(f"Subsampling {n_snapshots} snapshots to {len(indices)} frames")
    else:
        indices = list(range(n_snapshots))
    
    # Determine temperature range
    all_temps = []
    for idx in indices:
        snap = data['snapshots'][idx]
        all_temps.extend(snap['temperatures'])
    
    if temp_min is None:
        temp_min = min(all_temps)
    if temp_max is None:
        temp_max = max(all_temps)
    
    print(f"Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
    
    # Store original min/max for reset functionality
    original_temp_min = temp_min
    original_temp_max = temp_max
    absolute_temp_min = min(all_temps)
    absolute_temp_max = max(all_temps)
    
    # Create frames for animation
    frames = []
    
    # Scale dimensions for visualization (convert m to mm for better view)
    scale = 1000  # m to mm
    
    # Build first frame and all subsequent frames
    first_mesh_data = None
    
    for frame_idx, snap_idx in enumerate(indices):
        snap = data['snapshots'][snap_idx]
        positions = snap['positions']
        dimensions = snap['dimensions']
        temperatures = snap['temperatures']
        types = snap['types']
        
        # Create mesh for each node
        x_all, y_all, z_all = [], [], []
        i_all, j_all, k_all = [], [], []
        intensities = []
        hover_texts = []
        
        vertex_offset = 0
        for node_idx, (pos, dims, temp, node_type) in enumerate(zip(positions, dimensions, temperatures, types)):
            if np.all(dims == 0):
                continue
                
            vertices, faces = create_cube_mesh(pos * scale, dims * scale)
            
            x_all.extend(vertices[:, 0])
            y_all.extend(vertices[:, 1])
            z_all.extend(vertices[:, 2])
            
            # Add triangular faces (each quad = 2 triangles)
            for face in faces:
                i_all.extend([vertex_offset + face[0], vertex_offset + face[0]])
                j_all.extend([vertex_offset + face[1], vertex_offset + face[2]])
                k_all.extend([vertex_offset + face[2], vertex_offset + face[3]])
            
            # Color based on temperature (normalized)
            norm_temp = (temp - temp_min) / (temp_max - temp_min + 1e-6)
            intensities.extend([norm_temp] * 8)  # Same color for all 8 vertices
            
            hover_text = f"Node {node_idx}<br>Type: {node_type}<br>Temp: {temp:.1f}°C"
            hover_texts.extend([hover_text] * 8)
            
            vertex_offset += 8
        
        mesh_data = dict(
            x=x_all, y=y_all, z=z_all,
            i=i_all, j=j_all, k=k_all,
            intensity=intensities,
            text=hover_texts
        )
        
        if frame_idx == 0:
            first_mesh_data = mesh_data
        
        frames.append(go.Frame(
            data=[go.Mesh3d(**mesh_data, colorscale=colormap, cmin=0, cmax=1, hoverinfo='text')],
            name=str(snap_idx)
        ))
    
    # Create figure with first frame
    fig = go.Figure(
        data=[go.Mesh3d(
            **first_mesh_data,
            colorscale=colormap,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=dict(text='Temperature [°C]', side='right'),
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=[f'{temp_min:.0f}', f'{temp_min + (temp_max-temp_min)*0.25:.0f}',
                         f'{temp_min + (temp_max-temp_min)*0.5:.0f}', 
                         f'{temp_min + (temp_max-temp_min)*0.75:.0f}', f'{temp_max:.0f}']
            ),
            hoverinfo='text'
        )],
        frames=frames
    )
    
    # Create slider steps
    sliders = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            prefix="Time: ",
            visible=True,
            xanchor="center",
            suffix=" s"
        ),
        transition=dict(duration=0),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.05,
        y=0,
        steps=[
            dict(
                args=[[str(indices[i])], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                label=f"{data['timesteps'][indices[i]]:.1f}",
                method="animate"
            ) for i in range(len(indices))
        ]
    )]
    
    # Add play/pause buttons and temperature range controls
    updatemenus = [
        # Play/Pause buttons
        dict(
            type="buttons",
            showactive=False,
            y=0.15,
            x=0.05,
            xanchor="right",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                )
            ]
        ),
        # Temperature range presets
        dict(
            type="dropdown",
            showactive=True,
            y=0.3,
            x=1.15,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="Auto Range (Original)",
                    method="restyle",
                    args=[{
                        "colorbar.ticktext": [
                            [f'{original_temp_min:.0f}', 
                             f'{original_temp_min + (original_temp_max-original_temp_min)*0.25:.0f}',
                             f'{original_temp_min + (original_temp_max-original_temp_min)*0.5:.0f}', 
                             f'{original_temp_min + (original_temp_max-original_temp_min)*0.75:.0f}', 
                             f'{original_temp_max:.0f}']
                        ]
                    }]
                ),
                dict(
                    label=f"Full Range ({absolute_temp_min:.0f}–{absolute_temp_max:.0f}°C)",
                    method="restyle",
                    args=[{
                        "colorbar.ticktext": [
                            [f'{absolute_temp_min:.0f}', 
                             f'{absolute_temp_min + (absolute_temp_max-absolute_temp_min)*0.25:.0f}',
                             f'{absolute_temp_min + (absolute_temp_max-absolute_temp_min)*0.5:.0f}', 
                             f'{absolute_temp_min + (absolute_temp_max-absolute_temp_min)*0.75:.0f}', 
                             f'{absolute_temp_max:.0f}']
                        ]
                    }]
                ),
                dict(
                    label="Low Temp Focus (0–300°C)",
                    method="restyle",
                    args=[{
                        "colorbar.ticktext": [
                            ['0', '75', '150', '225', '300']
                        ]
                    }]
                ),
                dict(
                    label="Mid Temp Focus (0–600°C)",
                    method="restyle",
                    args=[{
                        "colorbar.ticktext": [
                            ['0', '150', '300', '450', '600']
                        ]
                    }]
                ),
                dict(
                    label="High Temp Focus (1000–1500°C)",
                    method="restyle",
                    args=[{
                        "colorbar.ticktext": [
                            ['1000', '1125', '1250', '1375', '1500']
                        ]
                    }]
                ),
                dict(
                    label="Interlayer Range (25–250°C)",
                    method="restyle",
                    args=[{
                        "colorbar.ticktext": [
                            ['25', '81', '138', '194', '250']
                        ]
                    }]
                ),
            ]
        )
    ]
    
    fig.update_layout(
        title=dict(text='WAAM Thermal Simulation - Node Temperatures<br><sub>Use dropdown (right) to adjust temperature scale</sub>', x=0.5),
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='data'
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        width=1200,
        height=800
    )
    
    # Save as HTML
    html_file = 'visualization.html'
    fig.write_html(html_file)
    print(f"Saved interactive visualization to {html_file}")
    
    # Show figure
    fig.show()


def visualize_with_matplotlib(data, colormap='hot', temp_min=None, temp_max=None):
    """
    Create 3D visualization using Matplotlib with slider.
    """
    print("Creating Matplotlib visualization...")
    
    # Determine temperature range
    all_temps = []
    for snap in data['snapshots']:
        all_temps.extend(snap['temperatures'])
    
    if temp_min is None:
        temp_min = min(all_temps)
    if temp_max is None:
        temp_max = max(all_temps)
    
    print(f"Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)
    
    # Colormap setup
    cmap = cm.get_cmap(colormap)
    norm = Normalize(vmin=temp_min, vmax=temp_max)
    
    # Scale for visualization
    scale = 1000  # m to mm
    
    # Current snapshot index
    current_idx = [0]
    
    def update_plot(idx):
        ax.clear()
        
        snap = data['snapshots'][idx]
        positions = snap['positions']
        dimensions = snap['dimensions']
        temperatures = snap['temperatures']
        types = snap['types']
        
        for pos, dims, temp, node_type in zip(positions, dimensions, temperatures, types):
            if np.all(dims == 0):
                continue
            
            vertices, faces = create_cube_mesh(pos * scale, dims * scale)
            
            # Create polygon collection for the faces
            face_vertices = [[vertices[v] for v in face] for face in faces]
            
            color = cmap(norm(temp))
            
            poly = Poly3DCollection(face_vertices, alpha=0.8)
            poly.set_facecolor(color)
            poly.set_edgecolor('black')
            poly.set_linewidth(0.3)
            ax.add_collection3d(poly)
        
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_title(f'WAAM Thermal Simulation - t = {data["timesteps"][idx]:.2f}s')
        
        # Auto-scale axes
        all_pos = positions * scale
        all_dims = dimensions * scale
        if len(all_pos) > 0:
            max_range = np.max([
                np.max(all_pos[:, 0] + all_dims[:, 0]/2) - np.min(all_pos[:, 0] - all_dims[:, 0]/2),
                np.max(all_pos[:, 1] + all_dims[:, 1]/2) - np.min(all_pos[:, 1] - all_dims[:, 1]/2),
                np.max(all_pos[:, 2] + all_dims[:, 2]/2) - np.min(all_pos[:, 2] - all_dims[:, 2]/2)
            ]) / 2
            
            mid_x = (np.max(all_pos[:, 0]) + np.min(all_pos[:, 0])) / 2
            mid_y = (np.max(all_pos[:, 1]) + np.min(all_pos[:, 1])) / 2
            mid_z = (np.max(all_pos[:, 2]) + np.min(all_pos[:, 2])) / 2
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        fig.canvas.draw_idle()
    
    # Initial plot
    update_plot(0)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Temperature [°C]')
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'Timestep', 0, len(data['snapshots']) - 1,
        valinit=0, valstep=1
    )
    
    def on_slider_change(val):
        idx = int(val)
        current_idx[0] = idx
        update_plot(idx)
    
    slider.on_changed(on_slider_change)
    
    # Add play/pause button
    ax_button = plt.axes([0.85, 0.05, 0.1, 0.03])
    button = Button(ax_button, 'Play')
    
    playing = [False]
    
    def toggle_play(event):
        playing[0] = not playing[0]
        button.label.set_text('Pause' if playing[0] else 'Play')
        
        if playing[0]:
            animate()
    
    def animate():
        if not playing[0]:
            return
        
        current_idx[0] = (current_idx[0] + 1) % len(data['snapshots'])
        slider.set_val(current_idx[0])
        
        fig.canvas.flush_events()
        plt.pause(0.05)
        
        if playing[0]:
            fig.canvas.manager.window.after(50, animate)
    
    button.on_clicked(toggle_play)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize WAAM thermal simulation node temperatures in 3D'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        default='node_temperatures.h5',
        help='Input HDF5 file path (default: node_temperatures.h5)'
    )
    parser.add_argument(
        '--colormap', '-c',
        type=str,
        default='hot',
        help='Colormap for temperature (default: hot). Options: hot, jet, viridis, plasma, etc.'
    )
    parser.add_argument(
        '--temp-min',
        type=float,
        default=None,
        help='Minimum temperature for color scale (default: auto)'
    )
    parser.add_argument(
        '--temp-max',
        type=float,
        default=None,
        help='Maximum temperature for color scale (default: auto)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['plotly', 'matplotlib', 'auto'],
        default='auto',
        help='Visualization backend (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.file}...")
    try:
        data = load_hdf5_data(args.file)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        print("Run log_node_temperatures.py first to generate the data file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(data['snapshots'])} timesteps")
    print(f"Simulation metadata: {data['metadata']}")
    
    # Choose backend
    backend = args.backend
    if backend == 'auto':
        if HAS_PLOTLY:
            backend = 'plotly'
        elif HAS_MATPLOTLIB:
            backend = 'matplotlib'
        else:
            print("Error: No visualization backend available.")
            print("Please install plotly or matplotlib:")
            print("  pip install plotly")
            print("  pip install matplotlib")
            sys.exit(1)
    
    # Visualize
    if backend == 'plotly':
        if not HAS_PLOTLY:
            print("Error: Plotly not installed. Install with: pip install plotly")
            sys.exit(1)
        visualize_with_plotly(data, args.colormap, args.temp_min, args.temp_max)
    else:
        if not HAS_MATPLOTLIB:
            print("Error: Matplotlib not installed. Install with: pip install matplotlib")
            sys.exit(1)
        visualize_with_matplotlib(data, args.colormap, args.temp_min, args.temp_max)


if __name__ == '__main__':
    main()
