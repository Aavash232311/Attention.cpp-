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
    float *A,      // (a, b, c, d)
    float *B,      // (a, b, d, e)
    float *C,      // (a, b, c, e)
    float scaling, // to re-use this for something like attention sore, and backpropagation.
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
    C[out_idx] = sum * scaling;
}

/*
Just for me to unfold the logic here.
J1 = diag(P[0]) - P[0]·P[0]T
J2 = diag(P[1]) - P[1]·P[1]T
J3 = diag(P[2]) - P[2]·P[2]T

The shape of matrix P is:  (batch, n_head, seq_len, seq_len) from the attn head.


I was hitting the so called "flow state" when I wrote parallel reduction for
forward pass kernels reading pdf's from NVIDA now I do not remember, but
lets try to unfold first.
*/

__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// One Kernel that accounts for seq_len > 32 if its small don't care
// even though we would have the advantage of warp level reduction.

// forget 1024 hardware limit for NOW at least lets get the model working atleast
// it will be a weak model but lets focus on getting the result right at first.
__global__ void softmaxBackTankKernel(
    float *P,  // (batch_size * num_heads * seq_len * seq_len )
    float *dY, // Shape (batch_size, seq_len, num_head, head_dim) again I might be wrong I am old.
    float *out,
    int N,
    int batch_size,
    int seq_len,
    int n_head)
{

    int batch_idx = blockIdx.z;
    int nhead_idx = blockIdx.y;
    int seq_len_idx1 = blockIdx.x;
    int seq_len_idx2 = threadIdx.x;

    int lane = threadIdx.x % 32;     // position within warp
    int warp_id = threadIdx.x / 32;  // position within block
    int num_warps = blockDim.x / 32; // total warps avalible

    int row_base = batch_idx * (n_head * seq_len * seq_len) + nhead_idx * (seq_len * seq_len) + seq_len_idx1 * seq_len;

    // I will note here, I am just learning, this will get populated and reduction happens in the warp level.
    __shared__ float smem_pdy[32]; // certian limit is there depending upon the GPU but this should be fine;.
    __shared__ float s_shared;

    // this is what I like to call surface level reduction
    // I have noted this concept on softmax_activation.org
    float tempSum = 0.0f;
    for (int i = seq_len_idx2; i < seq_len; i += blockDim.x)
    {
        // here i is the offset and blockDim.x is the number of thread in a block.
        tempSum += P[row_base + i] * dY[row_base + i];
    } // multipled with the upstream gradient dY, I like to call it G but, school damn.
    // I have a cheat sheet in collab somewhere.

    tempSum = warpReduceSum(tempSum);

    __syncthreads();

    if (seq_len_idx2 == 0)
    {
        // lane 0 if the each warp get assigned the reduced sum.
        smem_pdy[warp_id] = tempSum;
    } // confusing but I might get used to it, I promise, even after I wrote this from scratch twice already :)

    __syncthreads();

    float rowSum = (lane < num_warps) ? smem_pdy[lane] : 0.0f;
    if (warp_id == 0)
        rowSum = warpReduceSum(rowSum);

    if (seq_len_idx2 == 0)
        s_shared = rowSum;

    __syncthreads();

    float s = s_shared;

    if (seq_len_idx2 < seq_len)
    {
        for (int i = seq_len_idx2; i < seq_len; i += blockDim.x)
            out[row_base + i] = P[row_base + i] * (dY[row_base + i] - s);
    }
}

__device__ __forceinline__ void ParallelReducer(float &localSum)
{
    for (int offset = 16; offset > 0; offset /= 2)
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    localSum = __shfl_sync(0xffffffff, localSum, 0);
}

// LayerNorm Backpropagation

/*
    Eneginnering tradeoffs here, if we have that (x - u) from our forward pass kernel
    then we will need to reserve our VRAM, lets re-compute that again here.
*/

// C dimension > 32 we use shared memory here, if it was  < 32 then register could talk with each other in faster way, even if they are they wont because of this but its okay here.
// Note:- I am not so sharp and smart so I am taking my time here to derive and understand.
__global__ void LayerNormBackPropgationKernel(
    float *x, // (B, T, C)
    float *G, // (B, T, C)
    float *gamma,
    float *sigma,
    int D,
    int B,
    int T,
    int C)
{
    int batch_idx = blockIdx.y;
    int row_idx = blockIdx.x;

    const float *row = x + batch_idx * (T * C) + row_idx * C;
    float *out_row = x + batch_idx * (T * C) + row_idx * C;

    int lane = threadIdx.x % 32;     // position within warp, basically "one" thread within a cuda wrap
    int warp_id = threadIdx.x / 32;  // position within block
    int num_warps = blockDim.x / 32; // total warps avalible

    __shared__ float shared_sum[32];

    float sum = 0.0f;
    // partial sum here, for each thread
    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        sum += row[i];
    }

    // reduce within each wrap
    ParallelReducer(sum);

    // Now the sum is one single output of the sum within that wrap
    // and if the lane = 0 then we will assign that sum value of that wrap
    // to shared_sum by assigning wrap_id
    if (lane == 0)
    {
        shared_sum[warp_id] = sum;
    }

    __syncthreads();

    if (warp_id == 0)
    {
        sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        // reduce that element within that shared memory
        // when each wrap output is contributed
        ParallelReducer(sum);

        if (lane == 0)
            shared_sum[0] = sum;
    }

    __syncthreads();

    float mean = shared_sum[0] / C;

    float var_sum = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        float diff = row[i] - mean;
        var_sum += diff * diff;
    } // this I like to call reduction in surface

    ParallelReducer(var_sum);

    if (lane == 0)
    {
        shared_sum[warp_id] = var_sum;
    }

    __syncthreads();

    if (warp_id == 0)
    {
        var_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        ParallelReducer(var_sum);

        if (lane == 0)
        {
            shared_sum[0] = var_sum;
        }
    }

    __syncthreads();

    float variance = shared_sum[0] / C; // mean of that var * var
    float std = sqrtf(variance + 1e-8f);

    float epsilon = 1e-8f;

    // Here we have var, std dev, and mean
    // we just need to write the formula differently from that we derived in flashback.md

    // each thread will be touching here,
    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        // lets do parts by pars, due to the fact that me being not so smart.
        // flashattention.md documentation here I did the derivative in paper 
        // and worte them into .md file

        float first_comp = gamma[i] / D * sqrtf((std * std) + epsilon);

        float x_u = row[i] - mean;

        float second_comp = ((D - 1) - (x_u * x_u) / (std * std) + epsilon);

        out_row[i] = first_comp * second_comp * G[i];
    }
}

extern "C"
{
    void layerNormBackGradKerel()
    {

    }

    void softmaxBackGradKernel(
        float *P,
        float *dY,
        float *out,
        int N,
        int batch_size,
        int seq_len,
        int n_head)
    {
        dim3 blockSize(((seq_len + 31) / 32) * 32, 1, 1); // enough threads to cover one row, e.g. round seq_len up to nearest 32
        dim3 gridSize(
            seq_len,
            n_head,
            batch_size);

        softmaxBackTankKernel<<<gridSize, blockSize>>>(
            P, dY, out, N, batch_size, seq_len, n_head);
        cudaDeviceSynchronize();
    }

    void MatMul4D(
        float *A,      // (a, b, c, d)
        float *B,      // (a, b, d, e)
        float *C,      // (a, b, c, e)
        float scaling, // pass 1.0f if not scaling
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
            A, B, C, scaling,
            a, b, c, d, e);
        cudaDeviceSynchronize();
    }
}