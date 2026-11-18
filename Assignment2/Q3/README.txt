Naive version: 
files: run_gpu.c , matmul_gpu.cu
Running: 
	make run_gpu
	./run_gpu stories15M.bin -t 0


Optimizied version:
files: run_gpu_opt.c , matmul_gpu_opt.cu
Running:
	make run_gpu_opt
	./run_gpu_opt stories15M.bin -t 0
