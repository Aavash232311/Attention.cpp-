#include <iostream>
#include <iterator>
#include <math.h>
#include <cuda_runtime.h>

/*
We expect this to return a softmax function, for example x = [2, 1, 0]
softmax(x1) = e^2/e^2 + e^1 + e^0
softmax(x2) = e^1/e^2 + e^1 + e^0
softmax(x3) = e^0/e^2 + e^1 + e^0

*/

__global__ void softmax_kernel(float *arr, float *out, size_t N)
{
    __shared__ float total_sum; // memeory across all threads in the block

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadIdx.x == 0)
    {
        total_sum = 0.0f; // initially
    }

    __syncthreads(); // since thread are computing in parallel, forces all thread in a block to wait until everyone reaches the same line.

    if (i < N)
    {
        atomicAdd(&total_sum, expf(arr[i]));
    }

    __syncthreads();

    if (i < N)
    {
        out[i] = expf(arr[i]) / total_sum;
    }
}

__global__ void positional_embedding_kernel(float *dimension, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        if (i % 2 == 0)
        {
            printf("Even \n");
        }
        else
        {
            printf("Odd \n");
        }
    }
}

extern "C" void softmax(float *arr, float *out, int N)
{
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(arr, out, N);

    // wait for GPU to finish so the print statements display
    cudaDeviceSynchronize();
}

extern "C" void postionalEmbeddings(float *dimension, int N)
{
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    positional_embedding_kernel<<<blocks_per_grid, threads_per_block>>>(dimension, N);
}