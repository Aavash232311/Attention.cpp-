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
    int M,      // batch_size,
    int N,      // d_head
    int K,      // seq_len
    bool reverse = false)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x;
    int d = threadIdx.x;

    int currentIdx = b * (num_heads * K * head_dim) + h * (K * head_dim) + s * (head_dim) + d;

    int idxOut = b * (num_heads * head_dim * K) + h * (head_dim * K) + d * (K) + s;
    if (!(reverse))
        out[idxOut] = arr[currentIdx];
    else
        out[currentIdx] = arr[idxOut];
}

// Cross entropy tank
// we will fuse everything here.
__global__ void oneHotKernel(
    int *y,     // (B, T )
    float *out, // (B*T, vocab_size)
    int N,      // number of elements in y as if it was flat which it is.
    int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        // because you will be opreating under element of y
        int label = y[idx]; // this is the "position" that we want to be on
        out[idx * width + label] = 1.0f;
    }
}
// this y is typically (B, T)

// NOTE:- (B, T, vocab_size)
// they way I image this part is B, and T as a 2D matrix
// and then vocab size as a cell or like place holder inside of it
// SO: for our each B,T we will have a unique vocab as proballity.
__global__ void crossEntropyLoss(
    float *x,   // (B, T, vocab_size)
    int *y,     // (vocab_size)
    float *out, // N loss across all B,T basically.
    int seq_len,
    int vocab_size,
    int N)
{
    // H(p) = -p log(y)
    // The way we define the perfectness is 1 here.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // so here we basically want to read from the index of x to y and compute
    // the proballty

    if (idx < N)
    {
        int label = y[idx];                    // lookup psoition for x
        float p = x[idx * vocab_size + label]; // read from this x that vocab_size dimension
        out[idx] = -logf(fmaxf(p, 1e-12f));
    }

    // cross entropy is calcuated only for 1 position in the actual label
    // meaning we will have [0,0,0,...1,0,0] somewhere
    // for each B,T shape you have 1 thats B,T
}

extern "C"
{
    void CrossEntropy(
        float *x,         // (B, T, vocab_size)
        int *y,           // actual y (B, T) dimension
        float *oneHotOut, // (B, T, vocab_size)
        float *lossOut,
        int batch_size,
        int seq_len,
        int vocab_size)
    {
        int N = batch_size * seq_len;
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        dim3 grid(blocksPerGrid);
        dim3 block(threadsPerBlock);
        // actual (B, T) to one-hot encoded (B, T, vocab_size)
        oneHotKernel<<<grid, block>>>(y, oneHotOut, N, vocab_size);

        crossEntropyLoss<<<grid, block>>>(x, y, lossOut, seq_len, vocab_size, N);

        cudaDeviceSynchronize();
    }

    void TransposeKey(
        int num_heads,
        int head_dim,
        float *arr, //  Shape(batch_size, n_head, seq_len, d_head)
        float *out, // Shape (batch_size, n_head, head_dim, seq_len)
        int M,      // batch_size
        int N,      // d_head
        int K,      // seq_len
        bool reverse)
    {
        dim3 block(head_dim);
        dim3 grid(K, num_heads, M);

        TransposeKeyKernel<<<grid, block>>>(num_heads, head_dim, arr, out, M, N, K, reverse);

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
