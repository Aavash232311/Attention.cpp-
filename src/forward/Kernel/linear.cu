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

// Z = WX + B but this time in the GPU
// I am still learning to derive this sort of problem. If I can see the end-result then thats the defination of experience for me.

__global__ void WeightedSumKernel(
    float *x,
    float *w, // Shape(M, K)
    float *b, // Shape(K, N)
    float *c, // Shape(M, N)
    int M,
    int K,
    int N)
{

    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;

    if (rows >= M || cols >= N)
        return;

    float sum = 0.0f; // formula (row * width) + cols

    for (int row_b = 0; row_b < K; ++row_b)
    {
        float valA = x[(rows * K) + row_b];
        float valB = w[(row_b * N) + cols];
        sum += valA * valB;
    }

    c[(rows * N) + cols] = sum + b[cols];
}

__global__ void KaimingInitKernel(float *arr, curandState *state, int x, int y)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= x * y)
        return;

    // registers are ultra fast memory within SM's in the GPU
    curandState local = state[idx]; // global mem to register ex pos 0
    float std = sqrtf(2.0f / (float)x);
    arr[idx] = curand_normal(&local) * std; // local goes to pos 1

    // That opreation from global memory to register is physcially copied
    // back and fourh between the register and global memory
    state[idx] = local; // register -> global position 1, we need to write back the global because it is the only memory that survives after the Kernel ends.
    // if the each thread have different position on the global memory then why do we care about writing it back?
    // well the answer is if we using this KaimingInit again and the data is physically
}

// For token embeddings we did it on CPU because its initliized once the consturcotr is loaded for this we will be using the GPU
// curand_init is expensive and for each thread our performance will drop significantly

__global__ void SetUpRnd(curandState *state, unsigned long seed, int max_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //        seed, idx=which subsequent thread gets this, 0 = offset, and write in state[idx]

    if (idx >= max_threads)
        return;
    // Think of the global memory like a Queue.
    curand_init(seed, idx, 0, &state[idx]); // we can think of this as creating an instance of random class inside of global device memory.
}


extern "C"
{

    void KaimingInit(
        float *arr,
        curandState *state,
        int x,
        int y,
        unsigned long seed)
    {
        int total = x * y;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        SetUpRnd<<<blocks, threads>>>(state, seed, total);
        KaimingInitKernel<<<blocks, threads>>>(arr, state, x, y);
    }

    void WeightedSum(
        float *x, // Shape(M, K)
        float *w, // Shape(K, N)
        float *b, // Shape(N, )
        float *c, // Shape(M×N)
        int M,
        int K,
        int N)
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);

        WeightedSumKernel<<<grid, block>>>(x, w, b, c, M, K, N);

        cudaDeviceSynchronize();
    }

}