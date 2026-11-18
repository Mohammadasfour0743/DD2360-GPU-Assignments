#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d)
{
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= d)
        return;

    float sum = 0.0f;
    for (int j = lane; j < n; j += 32)
        sum += w[row * n + j] * x[j];

    for (int off = 16; off > 0; off /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }

    if (lane == 0)
        xout[row] = sum;
}

extern "C" void matmul(float *xout, float *x, float *w, int n, int d)
{
    float *d_x, *d_w, *d_xout;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_w, d * n * sizeof(float));
    cudaMalloc(&d_xout, d * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, d * n * sizeof(float), cudaMemcpyHostToDevice);

    int TPB = 256; // 8 warps per block
    int BPG = ((d * 32 + TPB - 1) / TPB);
    matmul_kernel<<<BPG, TPB>>>(d_xout, d_x, d_w, n, d);

    cudaMemcpy(xout, d_xout, d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_xout);
}