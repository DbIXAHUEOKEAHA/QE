import re
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import os

# File path
input_folder = '/home/kravt/QE/qe-7.4.1/BP/Monolayer/D = 0'
input_file = os.path.join(input_folder, 'ml_bp.nscf.out')

# Parse the file for k-points and energies
k_points = []
energies_list = []
with open(input_file, 'r') as f:
    current_k = None
    current_energies = []
    collecting_energies = False
    for line in f:
        # Detect k-point line
        if line.lstrip().startswith("k ="):
            nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', line)
            if len(nums) >= 3:
                # Save previous block if complete
                if current_k and len(current_energies) == 16:
                    k_points.append(current_k)
                    energies_list.append(current_energies)
                current_k = [float(x) for x in nums[:3]]
                current_energies = []
                collecting_energies = True
            continue
        # Collect energy values
        if collecting_energies and current_k:
            nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', line)
            if nums:
                current_energies.extend(float(x) for x in nums)
                if len(current_energies) >= 16:  # got enough bands
                    current_energies = current_energies[:16]
                    k_points.append(current_k)
                    energies_list.append(current_energies)
                    current_energies = []
                    collecting_energies = False

# Convert lists to NumPy arrays
k_points = np.array(k_points)  # shape (n_kpoints, 3)
energies = np.array(energies_list).T  # shape (n_bands, n_kpoints)
print(f"k_points shape: {k_points.shape}")
print(f"energies shape: {energies.shape}")

# Select 8th and 9th bands (Python indices 7 and 8) and shift by Fermi level

n_band_val = 9
n_band_cond = 10

band_val = energies[n_band_val, :] 
band_cond = energies[n_band_cond, :] 

fermi_energy = -3.6358


band_val -= fermi_energy
band_cond -= fermi_energy

# Apply region filter (around K-point: kx 0.23-0.43, ky 0.48-0.68)
mask = (
    (k_points[:, 0] >= -0.5) & (k_points[:, 0] <= 0.5) &
    (k_points[:, 1] >= -0.5) & (k_points[:, 1] <= 0.5)
)

kx = k_points[:, 0]
ky = k_points[:, 1]

kx_sel = k_points[mask, 0]
ky_sel = k_points[mask, 1]
band8_sel = band_val[mask]
band9_sel = band_cond[mask]

# Grid the data for 3D heatmap (interpolate to a regular grid)
grid_x, grid_y = np.mgrid[-0.5:0.5:500j, -0.5:0.5:500j]

# Create 3D heatmap with Plotly on a single canvas
fig = go.Figure()
alpha = 0.8

for i in np.arange(n_band_val, n_band_cond+1):
    band = energies[i, :] - fermi_energy
    grid_z = griddata((kx, ky), band, (grid_x, grid_y), method='cubic')
    
    # Add 3D surface for Band 8 (below Fermi level)
    fig.add_trace(
        go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Blues', 
                   opacity=alpha,  # 70% transparency
                   contours_z=dict(show=False, usecolormap=True, highlightcolor="navy", project_z=False,
                                   start = 20, end = -20, size = 1),
                   name='Band 8 (Below Fermi)',
                   coloraxis='coloraxis',
                   showscale=False)  # Disable individual colorbar
    )
    
    

data_raw = np.hstack([
    np.column_stack((kx, ky, band_val)),
    np.column_stack((kx, ky, band_cond))
])

np.savetxt(
    os.path.join(input_folder, 'ml_bp_band_full_BZ_coulumb.txt'),
    data_raw,
    header='kx_val ky_val energy_val kx_cond ky_cond energy_cond'
)

# Update layout with a single shared colorbar
fig.update_layout(
    title_text="3D Energy Heatmaps for BP (full BZ)",
    title_x=0.5,
    scene=dict(
        xaxis_title=r'kx',
        yaxis_title=r'ky',
        zaxis_title='Energy (eV)',
        aspectmode='cube',
        camera_eye=dict(x=1.5, y=1.5, z=1.0), # Adjust view angle
    ),
    coloraxis=dict(colorscale='RdBu', cmin=-3, cmax=3, colorbar_title="Energy (eV)"), # Shared colorbar
    width=800,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
    paper_bgcolor='rgba(245,245,245,1)', # Light gray background
    font=dict(size=14, color='black', family='Arial'),
    margin=dict(l=50, r=50, t=100, b=50),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Add lighting and interactivity
fig.update_traces(
    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1),
    lightposition=dict(x=100, y=100, z=1000)
)

# Show and save
fig.show()
fig.write_html(os.path.join(input_folder, 'ml_bp_3d_bands_single_coulumb.html'))