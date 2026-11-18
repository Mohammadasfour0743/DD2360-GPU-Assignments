import numpy as np
import os

def parse_vtk_scalars(filepath):
    """Robustly parses ASCII VTK files for both SCALARS and VECTORS data."""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    reading = False
    for line in lines:
        # Start reading data after LOOKUP_TABLE (scalars) or VECTORS definition
        if 'LOOKUP_TABLE' in line or 'VECTORS' in line:
            reading = True
            continue
        if reading:
            # Stop if we hit a new header section
            if any(kw in line for kw in ['DATASET', 'DIMENSIONS', 'ORIGIN', 'SPACING', 'POINT_DATA', 'SCALARS', 'VECTORS']):
                reading = False
                continue
            for token in line.split():
                try:
                    data.append(float(token))
                except ValueError:
                    pass
                    
    return np.array(data)

def compare_runs(cpu_dir, gpu_dir, filename):
    cpu_path = os.path.join(cpu_dir, filename)
    gpu_path = os.path.join(gpu_dir, filename)
    
    if not os.path.exists(cpu_path) or not os.path.exists(gpu_path):
        print(f"Skipping {filename}: File(s) not found.")
        return

    cpu_data = parse_vtk_scalars(cpu_path)
    gpu_data = parse_vtk_scalars(gpu_path)
    
    # Calculate error metrics
    abs_diff = np.abs(cpu_data - gpu_data)
    max_abs_err = np.max(abs_diff)
    mean_abs_err = np.mean(abs_diff)
    
    # Relative L2 error
    norm_cpu = np.linalg.norm(cpu_data)
    rel_l2 = np.linalg.norm(cpu_data - gpu_data) / norm_cpu if norm_cpu != 0 else 0.0

    print(f"--- Results for {filename} ---")
    print(f"  Max Absolute Error: {max_abs_err:.2e}")
    print(f"  Mean Absolute Error: {mean_abs_err:.2e}")
    print(f"  Relative L2 Error:   {rel_l2:.2e}\n")

# Run comparisons for key output files
files_to_check = ['E_10.vtk', 'B_10.vtk', 'rhoe_10.vtk', 'rhoi_10.vtk', 'rho_net_10.vtk']

for fname in files_to_check:
    compare_runs('cpu_output', 'gpu_output', fname)