#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// same kernel
__global__ void matmul_kernel(float *xout, const float *x, const float *w, int n, int d)
{
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= d)
        return;
    float sum = 0.0f;
    for (int j = lane; j < n; j += 32)
        sum += w[(size_t)row * n + j] * x[j];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0)
        xout[row] = sum;
}

static const float *h_base = NULL; // position of model in CPU
static float *d_base = NULL;       // position of model in GPU
static float *d_x = NULL;
static float *d_xout = NULL;

// transfer model from CPU to GPU one time
extern "C" void gpu_init(const float *h_data, size_t bytes, int max_n, int max_d)
{
    h_base = h_data;
    cudaMalloc(&d_base, bytes);
    cudaMemcpy(d_base, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_x, max_n * sizeof(float));
    cudaMalloc(&d_xout, max_d * sizeof(float));
}

extern "C" void gpu_free(void)
{
    cudaFree(d_base);
    cudaFree(d_x);
    cudaFree(d_xout);
}

extern "C" void matmul(float *xout, float *x, float *w, int n, int d)
{
    const float *d_w = d_base + (w - h_base); // translate CPU pointer to GPU pointer

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    int TPB = 256; // 8 warps per block
    int BPG = ((d * 32 + TPB - 1) / TPB);
    matmul_kernel<<<BPG, TPB>>>(d_xout, d_x, d_w, n, d);

    cudaMemcpy(xout, d_xout, d * sizeof(float), cudaMemcpyDeviceToHost);
}