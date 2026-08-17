#include <iostream>
#include <iterator>
#include <math.h>
#include <mma.h>
#include <random>
#include <vector>
#include <cfloat>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// we will experiement with tensor cors later on :)
__global__ void matmulLastTwo4DKernel(
    float *A, // (a, b, c, d)
    float *B, // (a, b, d, e)
    float *C, // (a, b, c, e)
    int a,
    int b,
    int c,
    int d,
    int e)
{
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;

    if (rows >= c || cols >= e)
        return;

    int c_a = blockIdx.z / b;
    int c_b = blockIdx.z % b;

    int skipA = c_a * (b * c * d) + c_b * (c * d);
    int skipB = c_a * (b * d * e) + c_b * (d * e);

    float sum = 0.0f;
    for (int i = 0; i < d; ++i)
    {
        /* 
            Little bit of a re-cap after my cooperate SWE work I might have forgotten these,
            skipA = normal offset, rows and cols are globally unqiue id, we need to skip those to reach the particular thread.

        */
        float valA = A[skipA + rows * d + i];
        float valB = B[skipB + i * e + cols];
        sum += valA * valB;
    }

    int out_idx = c_a * (b * c * e) + c_b * (c * e) + rows * e + cols;
    C[out_idx] = sum;
}

// How am I gonna be so god damn genious to distrubute the load
// of that formula derived in flashback.md row wise to do the softmax
// backpropagation. 


__global__ void softmaxBackTankKernel(

)
{
    
}


extern "C"
{
    void MatMul4D(
        float *A, // (a, b, c, d)
        float *B, // (a, b, d, e)
        float *C, // (a, b, c, e)
        int a,
        int b,
        int c,
        int d,
        int e)

    {
        dim3 blockDim(16, 16, 1);
        dim3 gridDim(
            (e + blockDim.x - 1) / blockDim.x, // cols = e
            (c + blockDim.y - 1) / blockDim.y, // rows = c
            a * b                              // combined batch*head axis
        );

        matmulLastTwo4DKernel<<<gridDim, blockDim>>>(
            A, B, C,
            a, b, c, d, e);
        cudaDeviceSynchronize();
    }
}