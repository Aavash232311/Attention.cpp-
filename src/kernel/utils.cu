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

__global__ void addVec(const float *a, const float *b, float *c, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

/*

ws = Shape(batch_size, seq_len, d_model)
            M,            N       K
This is a 3D tensor spins my head off lets compute this in this way.
we consider batch_size x seq_len = m,k

Shape ws(M, K)

new_shape = Shape(batch_size, seq_len, num_heads, head_dim)

ws = [
    batch_size:
           token 0: [..d_model]
           token 1: [..d_model]

    .... N batches
]

lets say we have 512 and N-head = 8


multi_headed = [
      batch size:     [8x64]
            token 0: [n_head][n_head][n_head]
            token 1: [n_head][n_head][n_head]
]

*/
// sig: ws, out, batch_size, seq_len, d_model, num_head, d_model
__global__ void multiHeadedAttentionKernel(
    float *ws,  // (B, T, C)
    float *out, //  B, n_head, seq_len, head_dim
    int batch_size,
    int seq_len,
    int d_model,
    int num_head,
    int head_dim)
{
    int token_idx = blockIdx.x; // which token (0 to M*N)
    int head_idx = blockIdx.y;  // which head
    int hd_idx = threadIdx.x;   // which elelemnt

    int batch_idx = token_idx / seq_len;
    int seq_idx = token_idx % seq_len;

    int idx = batch_idx * (seq_len * d_model) + seq_idx * (d_model) + head_idx * (head_dim) + hd_idx;
    // write this out to out, where it would be reshaped into multi headed attention
    int outIdx = batch_idx * (num_head * seq_len * head_dim) + head_idx * (seq_len * head_dim) + seq_idx * (head_dim) + hd_idx;

    out[outIdx] = ws[idx]; // proceed apporach
}

__global__ void TransposeKeyKernel(
    int num_heads,
    int head_dim,
    float *arr, // Shape (B, H, T, head_dim)
    float *out, // Shape (B, H, head_dim, T)
    int batch_size,      // batch_size, or whatever the least dim is
    int seq_len,      // seq_len
    bool reverse = false)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x;
    int d = threadIdx.x;

    int currentIdx = b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * (head_dim) + d;

    int idxOut = b * (num_heads * head_dim * seq_len) + h * (head_dim * seq_len) + d * (seq_len) + s;
    if (!(reverse))
        out[idxOut] = arr[currentIdx];
    else
        out[currentIdx] = arr[idxOut];
}

extern "C"
{

    void TransposeKey(
        float *arr, //  Shape(batch_size, n_head, seq_len, d_head)
        float *out, // Shape (batch_size, n_head, head_dim, seq_len)
        int num_heads,
        int head_dim,
        int batch_size, // batch_size
        int seq_len, // seq_len
        bool reverse)
    {
        dim3 block(head_dim);
        dim3 grid(seq_len, num_heads, batch_size);

        TransposeKeyKernel<<<grid, block>>>(num_heads, head_dim, arr, out, batch_size, seq_len, reverse);

        cudaDeviceSynchronize();
    }

    void multiHeadedAttention(
        int num_head,
        int head_dimension,
        float *ws,      // B, T, C
        float *out,     //  B, T, n_head, head_dim
        int batch_size, // batch_size
        int d_model,    // d_model
        int seq_len)    // seq_len, say we have n_head=8, head_dim=64, d_model=512
    {
        dim3 block(head_dimension);
        dim3 grid(batch_size * seq_len, num_head); // MXN will ignore btach_size, seq_len

        multiHeadedAttentionKernel<<<grid, block>>>(ws, out, batch_size, seq_len, d_model, num_head, head_dimension);

        cudaDeviceSynchronize();
    }
}
