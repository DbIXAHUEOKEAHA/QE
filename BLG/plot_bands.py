import re
import numpy as np
import matplotlib.pyplot as plt

# File paths
input_file = '/home/kravt/QE/qe-7.4.1/BP/Monolayer/D = 0/ml_bp.bands.out'
output_file = '/home/kravt/QE/qe-7.4.1/BP/Monolayer/D = 0/ml_bp_bands_data.txt'

# Initialize lists to store data
k_points = []
energies = []
bands_n = 20

# Parse the file
print(f"Opening file: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    print(f"Reading file content...")
    content = f.readlines()
    i = 0
    while i < len(content):
        line = content[i].strip()
        # Match k-point line (e.g., "k = 0.0000 0.0000 0.0000 ( 6957 PWs)   bands (ev):")
        k_match = re.match(r'k\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s*\(.*\)\s*bands\s*\(ev\)\s*:', line)
        if k_match:
            print(f"Matched k-point line {i}: {line}")
            current_k = [float(x) for x in k_match.groups()]  # kx, ky, kz
            current_energies = []
            i += 1  # Move to the next line for energies
            while i < len(content) and len(current_energies) < bands_n:
                energy_line = content[i].strip()
                print(f"Checking energy line {i}: {energy_line}")
                energies_match = re.findall(r'[-+]?\d+\.\d+(?:[eE][-+]?\d+)?', energy_line)
                if energies_match:
                    current_energies.extend([float(x) for x in energies_match])
                    print(f"Extracted energies: {current_energies}")
                i += 1
            if len(current_energies) == bands_n:
                print(f"Matched 16 energy values at lines {i-1} to {i}")
                k_points.append(current_k)
                energies.append(current_energies)
            else:
                print(f"Warning: Found only {len(current_energies)} energies at k-point {current_k}")
        else:
            i += 1

# Convert to numpy arrays
k_points = np.array(k_points)
energies = np.array(energies).T  # Transpose to have bands as rows, k-points as columns

# Save to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('# k-point index, kx, ky, kz (cartesian 2pi/alat), energies (eV) for 16 bands\n')
    f.write('# k-points follow path: Gamma -> K -> M -> Gamma (31 points)\n')
    for i, (k, e) in enumerate(zip(k_points, energies.T)):
        f.write(f'{i} {k[0]:.4f} {k[1]:.4f} {k[2]:.4f} ')
        f.write(' '.join(f'{x:.4f}' for x in e) + '\n')

print(f"Band data saved to {output_file}")
print(f"Number of k-points: {len(k_points)}, Number of bands: {len(energies)}")

# Plot the bands
if len(k_points) > 0:
    k_path = np.arange(len(k_points))
    plt.figure(figsize=(10, 6))
    for band in range(energies.shape[0]):
        plt.plot(k_path, energies[band], '-o', markersize=2)
    plt.xlabel('k-point index')
    plt.ylabel('Energy (eV)')
    plt.title('Band Structure')
    plt.grid(True)
    plt.xticks(ticks=np.arange(0, len(k_points), 5), labels=[f'{i}' for i in range(0, len(k_points), 5)])
    plt.show()
else:
    print("No data to plot.")