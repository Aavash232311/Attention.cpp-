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

#include "../interface_back.hpp"


// Derivation interfaceback.md
__global__ void upstream_dl_dz_kernel(
    float *actual, // (B ,T, vocab_size)
    float *predicted,
    float *delta,
    int B,
    int T,
    int vocab_size)
{
    int batch_idx = blockIdx.z;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= B || row_idx >= T || col_idx >= vocab_size)
        return;
    // we take delta from the last two elements only
    int idx = batch_idx * (T * vocab_size) + row_idx * vocab_size + col_idx;

    delta[idx] = predicted[idx] - actual[idx];
}

__global__ void transpose_last_two_kernel(float *input, float *output, int B, int T, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;

    if (idx < total)
    {
        int b = idx / (T * C);
        int remainder = idx % (T * C);
        int t = remainder / C;
        int c = remainder % C;

        int inputIdx = b * (T * C) + t * C + c;  // (B, T, C)
        int outputIdx = b * (C * T) + c * T + t; // (B, C, T)

        output[outputIdx] = input[inputIdx];
    }
}

__global__ void dl_dw_upstream_kernel(
    float *h_t,   // (B, C, T)          transposed h
    float *delta, // (B, T, vocab_size)
    float *out,   // (B, C, vocab_size)
    int B,
    int T,
    int C,
    int vocab_size)
{
    int batch_idx = blockIdx.z;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y; // C
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x; // vocab_size

    if (batch_idx >= B || row_idx >= C || col_idx >= vocab_size)
        return;

    float sum = 0.0f;
    for (int k = 0; k < T; ++k)
    {
        // transposne shape (B, C, T) escape C, T
        // delta shape (B, T, vocab_size)
        int idx_ht = (C * T) * batch_idx + T * row_idx + k;
        int idx_delta = (T * vocab_size) * batch_idx + vocab_size * k + col_idx;
        sum += h_t[idx_ht] * delta[idx_delta];
    }

    int idx_out = (C * vocab_size) * batch_idx + vocab_size * row_idx + col_idx;
    out[idx_out] = sum;
}

__global__ void wt_upstream_gradient_kernel(
    float *w,  // (d_model, vocab_size)
    float *wt, // (vocab_size, d_model)
    int d_model,
    int vocab_size)
{
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

     if (row_idx >= vocab_size || col_idx >= d_model)
        return;

    int out_idx = (vocab_size) * col_idx + row_idx;
    int idx_wt = row_idx * (d_model)  + col_idx;

    wt[idx_wt] = w[out_idx];
}

extern "C"
{
    // -------------- Backpropagation kernel wrappers -----------------
    void wt_upstream(
        float *w,
        float *wt,
        int d_model,
        int vocab_size)
    {
        dim3 block(16, 16);
        dim3 grid(
            (d_model + block.x - 1) / block.x,
            (vocab_size + block.y - 1) / block.y                                 
        );

        wt_upstream_gradient_kernel<<<grid, block>>>(
            w,
            wt,
            d_model,
            vocab_size
        );

        cudaDeviceSynchronize(); // wait :)
    }

    void dl_dw_upstream(
        float *h_t,
        float *delta,
        float *out,
        int B,
        int T,
        int C,
        int vocab_size)
    {
        // 1024 is the hardware limit and vocab_size can get large over time.
        // 16x16 per block
        dim3 block(16, 16, 1); // block.x handles part of vocab_size, block.y handles part of C

        dim3 grid(
            (vocab_size + block.x - 1) / block.x, // grid.x: enough blocks to cover all of vocab_size
            (C + block.y - 1) / block.y,          // grid.y: enough blocks to cover all of C
            B                                     // grid.z: one per batch element
        );

        dl_dw_upstream_kernel<<<grid, block>>>(h_t, delta, out, B, T, C, vocab_size);

        cudaDeviceSynchronize();
    }

    // Gradient flow in the LM head from (B, T, C) to (B, T, vocab_size) as a proballity score
    void lm_head_transpose_h(
        float *h, // (B, T, d_model)
        float *out,
        int B,
        int T,
        int C)
    {
        int threads = 256;
        int blocks = (B * T * C + 255) / 256;

        transpose_last_two_kernel<<<blocks, threads>>>(h, out, B, T, C);

        cudaDeviceSynchronize();
    }

    void upstream_dl_dz(
        float *actual,
        float *predicted,
        float *delta,
        int B,
        int T,
        int C)
    {
        dim3 block(16, 16, 1); // tune as needed block.x*block.y should be <=1024
        dim3 grid(
            (C + block.x - 1) / block.x, // col_idx / vocab_size
            (T + block.y - 1) / block.y, // row_idx / T
            B                            // batch_idx / B
        );
        upstream_dl_dz_kernel<<<grid, block>>>(actual, predicted, delta, B, T, C);
        cudaDeviceSynchronize();
    }
}