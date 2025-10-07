import numpy as np
import matplotlib.pyplot as plt
from pythtb import w90
import multiprocessing as mp
import time
import os
import functools
import pandas as pd
try:
    from tqdm import tqdm  # For progress bar
    use_tqdm = True
except ImportError:
    use_tqdm = False  # Fallback if tqdm is not installed

# Top-level function for eigenvalue solving (to fix pickling for multiprocessing)
def solve_one_k(tb, k):
    return tb.solve_one(k)

# User-defined parameters
folder = "/home/kravt/QE/qe-7.4.1/simul"  # Linux-style path for WSL
prefix = "blg_05"  # Prefix of Wannier90 files
nk_per_dim = 240  # Number of k-points per dimension (e.g., 30x30=900 for 2D)
force_2d = True  # Force 2D sampling (kx-ky plane at kz=0) for quasi-2D systems
sigma = 0.0025  # Gaussian broadening width in eV
ne = 2500  # Number of energy points

def compute_dos():
    # Start timing the entire script
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting DOS calculation...")

    # Check if required Wannier90 files exist
    required_files = [f"{prefix}.win", f"{prefix}_hr.dat", f"{prefix}_centres.xyz"]
    for file in required_files:
        file_path = os.path.join(folder, file)
        if not os.path.isfile(file_path):
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: File {file_path} not found. Please check the folder and prefix.")
            return

    # Load the Wannier90 model
    print(f"[{time.strftime('%H:%M:%S')}] Loading Wannier90 model from {folder}/{prefix}...")
    try:
        w = w90(folder, prefix)
        print(f"[{time.strftime('%H:%M:%S')}] Wannier90 files loaded.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Failed to load Wannier90 model: {e}")
        return

    # Create the tight-binding model
    tb = w.model()  # Add min_hopping_norm=1e-6 if needed for sparsity
    print(f"[{time.strftime('%H:%M:%S')}] Tight-binding model initialized. K-space dimensionality: {tb._dim_k}")
    if tb._dim_k != 2:
        print(f"[{time.strftime('%H:%M:%S')}] WARNING: Expected 2D system (e.g., bilayer graphene), but model has {tb._dim_k}D k-space. Check {prefix}.win.")

    # Generate uniform k-mesh (force 2D sampling if requested for 3D models)
    if force_2d and tb._dim_k == 3:
        print(f"[{time.strftime('%H:%M:%S')}] Forcing 2D sampling (kx-ky grid at kz=0) for quasi-2D system.")
        n_mesh = [nk_per_dim, nk_per_dim, 1]
    else:
        n_mesh = [nk_per_dim] * tb._dim_k
    print(f"[{time.strftime('%H:%M:%S')}] Generating k-mesh with dimensions {n_mesh} (total {np.prod(n_mesh)} k-points)...")
    kpts = tb.k_uniform_mesh(n_mesh)
    print(f"[{time.strftime('%H:%M:%S')}] K-mesh generated with {len(kpts)} k-points.")

    # Prepare partial function for multiprocessing (binds tb to solve_one_k)
    solve_func = functools.partial(solve_one_k, tb)

    # Parallelize eigenvalue computation with progress tracking
    n_cores = mp.cpu_count()
    print(f"[{time.strftime('%H:%M:%S')}] Computing eigenvalues using {n_cores} CPU cores...")
    try:
        with mp.Pool(processes=n_cores) as pool:
            if use_tqdm:
                evals_list = list(tqdm(pool.imap(solve_func, kpts), total=len(kpts), desc="Eigenvalues"))
            else:
                evals_list = pool.map(solve_func, kpts)  # No progress bar if tqdm unavailable
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Multiprocessing failed: {e}")
        print(f"[{time.strftime('%H:%M:%S')}] Falling back to serial eigenvalue computation...")
        if use_tqdm:
            evals_list = [solve_func(k) for k in tqdm(kpts, desc="Eigenvalues")]
        else:
            evals_list = [solve_func(k) for k in kpts]

    evals = np.array(evals_list)  # Shape: (n_kpts, n_bands)
    print(f"[{time.strftime('%H:%M:%S')}] Eigenvalues computed. Shape: {evals.shape}")

    # Flatten the eigenvalues array
    evals_flat = evals.flatten()
    print(f"[{time.strftime('%H:%M:%S')}] Eigenvalues flattened ({len(evals_flat)} total).")

    # Determine energy range for DOS
    emin = evals_flat.min() - 2 * sigma
    emax = evals_flat.max() + 2 * sigma
    print(f"[{time.strftime('%H:%M:%S')}] Energy range: [{emin:.3f}, {emax:.3f}] eV")

    # Create energy grid
    energy = np.linspace(emin, emax, ne)
    print(f"[{time.strftime('%H:%M:%S')}] Energy grid created with {ne} points.")

    # Vectorized DOS computation
    print(f"[{time.strftime('%H:%M:%S')}] Computing DOS with Gaussian broadening (sigma={sigma} eV)...")
    diff = energy[:, None] - evals_flat[None, :]
    gauss = np.exp(- (diff ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    dos = np.sum(gauss, axis=1)  # Sum over eigenvalues
    dos /= len(kpts)  # Normalize by number of k-points
    print(f"[{time.strftime('%H:%M:%S')}] DOS computed.")

    df = pd.DataFrame({'E, eV': energy, 'DOS': dos})
    df.to_csv(os.path.join(folder, 'DOS_TB.csv'), index = False)

    # Total runtime
    total_time = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Calculation complete. Total time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    compute_dos()