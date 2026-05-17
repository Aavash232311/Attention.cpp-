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

__global__ void softmax(float *arr, float *out, size_t N)
{
    __shared__ float total_sum; // memeory across all threads in the block

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadIdx.x == 0) {
        total_sum = 0.0f; // initially
    }

    __syncthreads(); // since thread are computing in parallel, forces all thread in a block to wait until everyone reaches the same line.

    if (i < N) {
        atomicAdd(&total_sum, expf(arr[i]));
    }

    __syncthreads();

    if (i < N) {
        out[i] = expf(arr[i]) / total_sum;
    }
}
