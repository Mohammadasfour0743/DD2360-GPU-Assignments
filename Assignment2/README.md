# Assignment 2: Execution Guide

This guide outlines the steps required to set up the environment, execute the CUDA programs, and reproduce the profiling and simulation outputs for **Problems 1–3** and the **Bonus** problem.

## Problem 1

- Open Google Colab, switch the runtime hardware accelerator to an **NVIDIA T4 GPU**, and copy the notebook contents into your workspace.
- Run the notebook cells sequentially from top to bottom to compile and execute the baseline CUDA implementation.
- Locate the tile size variable in the source code and modify its value to test different configurations, re-running the compilation and execution cells for each tile size.
- Run the profiling commands in the notebook to capture execution metrics using NVIDIA Nsight Compute.
- Download the generated profiling report files to your local machine and open them in the **NVIDIA Nsight Compute** GUI to inspect the results.

## Problem 2

## Problem 3

- Copy the 5 new files into the llama2 repository, and make sure the folder has the "stories15M.bin" data.

- Naive version: \
files: run_gpu.c , matmul_gpu.cu \
Running: \
	`make run_gpu` \
	`./run_gpu stories15M.bin -t 0`


- Optimizied version: \
files: run_gpu_opt.c , matmul_gpu_opt.cu \
Running: \
	`make run_gpu_opt` \
	`./run_gpu_opt stories15M.bin -t 0` \


## Bonus

- Open Google Colab with an **NVIDIA T4 GPU** runtime and copy the provided notebook cells into your environment.
- Clone the project repository and create a `data` directory inside the main `iPIC3D-mini` folder where the simulation will output its `.vtk` result files.
- Update the `Makefile` and `src/Particles.cu` using the modified code provided in the notebook (setting the architecture flag to `-arch=sm_75` and adding the GPU particle mover implementation).
- Compile with `make`, run the simulation with `GEM_2D.inp`, and download the generated `.vtk` files from the `data` folder.
- Modify `src/sputniPIC.cpp` again to call the GPU mover (`mover_PC_gpu`), recompile with `make`, re-run the simulation, and download the new `.vtk` files.
- Place the downloaded CPU and GPU result files into separate local folders matching the directory names expected in the Python checker script, then run the script to verify the outputs.
