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

__global__ void ReformShapeKernel(
    float *arr, // [batch_size, T, n_head, d_head]
    float *out, // (B, T, C)
    int batch_size,
    int seq_len,
    int d_model,
    int num_head,
    int head_dim)
{
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int hd_idx = threadIdx.x;

    int idx = batch_idx * (seq_len * num_head * head_dim) +
              seq_idx * (num_head * head_dim) + head_idx * (head_dim) + hd_idx;

    // d_model = num_head * head_dim
    // (B, T, num_head * head_dim)

    int c_idx = head_idx * head_dim + hd_idx;
    int outIdx = batch_idx * (seq_len * d_model) + seq_idx * (d_model) + c_idx;
    out[outIdx] = arr[idx];
}

// sweet and simple here because I am stil learning
// i want to get to the end result first then benchmark and
// see if we can use tensor cores here.
// QK should go through sqrt(d_model) at least here

/*
    idx = a * (B * C * D)
    + b * (C * D)
    + c * (D)
    + d
*/
__global__ void QKVMatmulKernel(
    float *QK,  // Shape(batch_size, n_head, T, T)
    float *V,   // Shape(batch_size, n_head, T, d_head)
    float *out, // (batch_size, n_head, T, d_head)
    int seq_len,
    int d_head,
    int n_head,
    int batch_size)
{
    int batch_idx = blockIdx.z;
    int nhead_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int dhead_idx = threadIdx.x;

    // we need to land on T T
    int QK_base = batch_idx * (n_head * seq_len * seq_len) + nhead_idx * (seq_len * seq_len);

    int v_base = batch_idx * (n_head * seq_len * d_head) + nhead_idx * (seq_len * d_head);

    float sum = 0.0f;

    for (int rowb = 0; rowb < seq_len; ++rowb)
    {
        // that QK_base and v_base are just offset so that we skip
        float valA = QK[QK_base + seq_idx * seq_len + rowb];
        float valB = V[v_base + rowb * d_head + dhead_idx];
        sum += valA * valB;
    }

    int out_idx = batch_idx * (n_head * seq_len * d_head) + nhead_idx * (seq_len * d_head) + seq_idx * d_head + dhead_idx;
    out[out_idx] = sum;
}

/*
    Note:- so that I do not get lost.
    We have (seq_len seq_len) at the end which is the result of matrix multiplication from
    QK^T matrix. A = Shape(batch_size, n_head, seq_len, head_dim), B = Shape(batch_size, n_head, head_dim, seq_len)
    C = (batch, n_head, seq_len, seq_len) it is multiplied by head_dim which is from the embeddings from that higher dimensional vector.
    So we are masking that last two seq_len,seq_len overtime I might find it weird on why we are masking this.

    Example of masking.
       (before mask)          mask              s (after masked_fill)
    [ 0.5  0.8  0.3  0.9 ]   [ 1 0 0 0 ]     [ 0.5  -1e9  -1e9  -1e9 ]
    [ 0.2  0.6  0.1  0.7 ]   [ 1 1 0 0 ]     [ 0.2   0.6  -1e9  -1e9 ]
    [ 0.9  0.3  0.4  0.2 ]   [ 1 1 1 0 ]     [ 0.9   0.3   0.4  -1e9 ]
    [ 0.1  0.5  0.8  0.6 ]   [ 1 1 1 1 ]     [ 0.1   0.5   0.8   0.6 ]

    (0, 1) 1 > 0 true then mask it
    (3, 2) 2 > 3 false do not mask it

    idx = a * (B * C * D)
    + b * (C * D)
    + c * (D)
    + d
*/

__global__ void UpperTriangularMaskingKernel(
    float *arr, // (batch, n_head, seq_len, seq_len)
                // (1, 1, T, T) so only two seq_len at the last are masked
    float val,
    int batch_size,
    int n_head,
    int seq_len)
{
    int batch_idx = blockIdx.z;
    int nhead_idx = blockIdx.y;
    int seq_len_idx1 = blockIdx.x;  // row
    int seq_len_idx2 = threadIdx.x; // cols

    // we care only about masking the (T, T) shape at the end. I really cannot visuize 4D in my head
    // If I sit with calculator manually flattening these matrix then using this formula I will land in the
    // correct place.

    int idx = batch_idx * (n_head * seq_len * seq_len) + nhead_idx * (seq_len * seq_len) + seq_len_idx1 * (seq_len) + seq_len_idx2;

    // so logic here is if this is row 0 in the TXT shape then after col zero every other value is masked.
    // and if this is row 1 then  till col 1 it is unmaked else every other value is masked.

    if (seq_len_idx2 > seq_len_idx1)
    {
        arr[idx] = val;
    }
}

/*
We expect this to return a softmax function, for example x = [2, 1, 0]
softmax(x1) = e^2/e^2 + e^1 + e^0
softmax(x2) = e^1/e^2 + e^1 + e^0
softmax(x3) = e^0/e^2 + e^1 + e^0

*/

// This default accounts for vocab_size > 32 which is likely the case in almost everytime.

/*
    First of all in our simple kernel we could talk between the warps.
    Now for something like vocab_size the problem is likely to be larger than 32.

    Let us take an example of maxed out kernel.

    Each block will have 1024 thread that is the hardware limit.
*/

// warp level reduction for the sum

__device__ __forceinline__ void warpReducerHelper(float &m, float &d)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other_m = __shfl_down_sync(0xffffffff, m, offset);
        float other_d = __shfl_down_sync(0xffffffff, d, offset);

        float m_new = fmaxf(m, other_m);
        d = d * __expf(m - m_new) + other_d * __expf(other_m - m_new);
        m = m_new;
    }
}

__global__ void SoftmaxKernel2D(
    float *arr, // (B, T, vocab_size)
    float *out,
    int batch_size,
    int seq_len,
    int vocab_size)
{
    int batch_idx = blockIdx.y;
    int row_idx = blockIdx.x;

    int col = threadIdx.x;
    if (col >= vocab_size)
        return;

    const float *row = arr + batch_idx * (seq_len * vocab_size) + row_idx * vocab_size;
    float *out_row = out + batch_idx * (seq_len * vocab_size) + row_idx * vocab_size;

    int lane = threadIdx.x % 32;     // position within warp
    int warp_id = threadIdx.x / 32;  // position within block
    int num_warps = blockDim.x / 32; // total warps avalible

    // I believe this is the online softmax activation that we were doing back then
    // seeing it after a long time.

    __shared__ float smem_m[32];
    __shared__ float smem_d[32];

    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) // jump to number of thread in a block.
    {
        float val = row[i];
        float m_new = fmaxf(local_m, val);
        local_d = local_d * __expf(local_m - m_new) + __expf(val - m_new);
        local_m = m_new;
    }

    // parallel reduction happenning inside of the kernel.
    warpReducerHelper(local_m, local_d);

    if (lane == 0)
    {
        smem_m[warp_id] = local_m;
        smem_d[warp_id] = local_d;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        local_m = (lane < num_warps) ? smem_m[lane] : -FLT_MAX;
        local_d = (lane < num_warps) ? smem_d[lane] : 0.0f;

        warpReducerHelper(local_m, local_d);

        if (lane == 0)
        {
            smem_m[0] = local_m;
            smem_d[0] = local_d;
        }
    }
    __syncthreads();

    float global_m = smem_m[0];
    float global_d = smem_d[0];

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x)
    {
        out_row[i] = __expf(row[i] - global_m) / global_d;
    }
}

// Only the last two batch are multiplied here rest of them are taken along
__global__ void QKmatmulKernel(
    float *Q,   // Shape(batch_size, n_head, seq_len, head_dim)
    float *Kt,  // Shape(batch_size, n_head, head_dim, seq_len)
    float *out, // (batch, n_head, seq_len, seq_len)
    int M,      // (seq_len, head_dim) = (M, N)
    int N,      // (head_dim, seq_len) = (N, M)
    int n_head)
{
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;

    if (rows >= M || cols >= M)
        return;

    int b = blockIdx.z / n_head;
    int h = blockIdx.z % n_head;

    float sum = 0.0f;
    int skipQ = b * (n_head * M * N) + h * (M * N); // idx after skipping the first two tensors
    int skipKt = b * (n_head * N * M) + h * (N * M);

    for (int k = 0; k < N; ++k)
    {
        // offset because we do not account for first two and then (rows * width) + cols
        float valA = Q[skipQ + rows * N + k];
        float valB = Kt[skipKt + k * M + cols];
        sum += valA * valB;
    }
    int out_offset = b * (n_head * M * M) + h * (M * M);
    out[out_offset + rows * M + cols] = sum;
}

// we can bind this into the same kernel but in order to build my thinking I am doing this.
// I must learn to derive a problem, thats what I call true understanding. Even if I forget the synatax.
__global__ void ScalerDvisionDModelKernel(
    float *arr, // (batch, n_head, seq_len, seq_len)
    int total_elem,
    float scaler,
    int n_head,
    int seq_len)
{
    int batch_idx = blockIdx.z;
    int nhead_idx = blockIdx.y;
    int seq_len_idx1 = blockIdx.x;  // row
    int seq_len_idx2 = threadIdx.x; // cols

    int idx = batch_idx * (n_head * seq_len * seq_len) + nhead_idx * (seq_len * seq_len) + seq_len_idx1 * (seq_len) + seq_len_idx2;

    if (idx < total_elem)
    {
        arr[idx] = arr[idx] / scaler;
    } // I see potentional of Kenrel Fusion here but I want to learn cuda so I am writing a different kernel to make my mind usedto it.
}

// works fine for seq_len < 32
__global__ void softmaxKrenel4D(
    float *arr, // (batch, n_head, seq_len, seq_len)
    float *out,
    int N,
    int seq_len,
    int n_head)
{

    int batch_idx = blockIdx.z;
    int nhead_idx = blockIdx.y;
    int seq_len_idx1 = blockIdx.x;  // row
    int seq_len_idx2 = threadIdx.x; // cols 0-31 thread in a wrap

    if (seq_len_idx2 >= seq_len)
        return;

    int idx = batch_idx * (n_head * seq_len * seq_len) + nhead_idx * (seq_len * seq_len) + seq_len_idx1 * seq_len + seq_len_idx2;

    float val = (idx < N) ? arr[idx] : -FLT_MAX;

    // wrap level reduction, because shared memory is little costly here.
    for (int offset = 16; offset > 0; offset /= 2)                 // half half half and max is carried along I said in myyy wayyy
        val = max(val, __shfl_down_sync(0xffffffff, val, offset)); // val and which thread do I want to read.
    float max_val = __shfl_sync(0xffffffff, val, 0);               // max in warp, its not an array to be mistaken each thread has its own register to store the value.

    // exp in register
    float exp_val = (idx < N) ? expf(arr[idx] - max_val) : 0.0f;

    // wrap level sum exp
    for (int offset = 16; offset > 0; offset /= 2)
        exp_val += __shfl_down_sync(0xffffffff, exp_val, offset);
    float sum_val = __shfl_sync(0xffffffff, exp_val, 0); // sum in warp

    // final formula
    if (idx < N)
        out[idx] = expf(arr[idx] - max_val) / sum_val;
}

/*
    Lets trace the parallel reduction here.

    we have a warp of 6 threads from 0 to 15 index.
    Basically yhis wrap talks with the thread registers the fastest possible memory in the GPU.

    Lets consider this thread
    Lane:   0  1  2  3  4  5  6  7
    Value:  1  2  3  4  5  6  7  8


    at level 1, we have offset = 4
    Lane 0: 1 + 5 =  6
    Lane 1: 2 + 6 = 8
    Lane 2: 3 + 7 = 10
    Lane 3: 4 + 8 = 12

    Result = [6, 8, 10, 12, 5, 6, 7, 8]

    level 2 offset = 2

    Level 0: = 6 + 10 = 16
    Level 1: = 8 + 12 = 20

    Result = [16, 20, 10, 12, 5, 6, 7, 8]

    Level 0 : 16 + 20 = 36

    [36, 20, 10, 12, 5, 6, 7, 8]

*/

__global__ void vectorAddKernel(const float *B, const float *C, float *A, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        A[i] = B[i] + C[i];
    }
}

/*
if you have Shape(A, B, C, D)

the formula to land on current elemenet is
idx = a * (B * C * D)
    + b * (C * D)
    + c * (D)
    + d
To get coordinate of that in a tensor
int a = idx / (B * C * D);
int b = (idx / (C * D)) % B;
int c = (idx / D) % C;
int d = idx % D;

Draw something flat, sit down with a calculator it will make sense.
It spins my head sometimes but thats it.

multi_headed = [
      batch size:     [8x64]
            token 0: [n_head][n_head][n_head]
            token 1: [n_head][n_head][n_head]
]
*/
// we can say that these kernel function are not flexible because it is hardcoded for each case, I know the fact but our goal is to understand this as much as possible.
// sig (num_heads, head_dim, arr, out, batch_size, seq_len, reverse);
__global__ void TransposeKernel(
    int num_head,
    int head_dim,
    float *arr,     // Shape(batch_size, num_heads, seq_len, head_dim)
    float *out,     // Shape(batch_size, seq_len, num_head, head_dim)
    int batch_size, // batch_size
    int seq_len,
    bool reverse = true)
{
    int token_idx = blockIdx.x; // which token (0 to M*N)
    int head_idx = blockIdx.y;  // which head
    int hd_idx = threadIdx.x;   // which elelemnt

    int batch_idx = token_idx / seq_len;
    int seq_idx = token_idx % seq_len;

    int idx = batch_idx * (seq_len * num_head * head_dim) + seq_idx * (num_head * head_dim) + head_idx * (head_dim) + hd_idx;

    int outIdx = batch_idx * (num_head * seq_len * head_dim) + head_idx * (seq_len * head_dim) + seq_idx * (head_dim) + hd_idx;

    if (reverse)
        out[idx] = arr[outIdx];
    else
        out[outIdx] = arr[idx];
}

extern "C"
{
    void ReformShapeWapper(
        float *arr, // [batch_size, T, n_head, d_head]
        float *out, // (B, T, C)
        int batch_size,
        int seq_len,
        int d_model,
        int num_head,
        int head_dim)
    {
        dim3 block(head_dim);
        dim3 grid(num_head, seq_len, batch_size);

        ReformShapeKernel<<<grid, block>>>(arr, out, batch_size, seq_len, d_model, num_head, head_dim);

        cudaDeviceSynchronize();
    }

    void QKVMatmulFinal(
        float *QK,
        float *V,
        float *out,
        int seq_len,
        int d_head,
        int n_head,
        int batch_size)
    {
        dim3 grid(seq_len, n_head, batch_size); // x=seq_len, y=n_head, z=batch_size
        dim3 block(d_head);
        QKVMatmulKernel<<<grid, block>>>(QK, V, out, seq_len, d_head, n_head, batch_size);

        cudaDeviceSynchronize();
    }
    void UpperTriangularMasking(
        float *arr, // Shape(batch_size, n_head, seq_len, seq_len)
        float val,
        int batch_size,
        int n_head,
        int seq_len)
    {
        //     blocIdx.x  blockIdx.y blockIdx.z
        dim3 grid(seq_len, n_head, batch_size);
        dim3 block(seq_len);

        UpperTriangularMaskingKernel<<<grid, block>>>(arr, val, batch_size, n_head, seq_len);

        cudaDeviceSynchronize();
    }

    void QKmatmul(
        float *Q,
        float *Kt,
        float *out,
        int M,
        int N,
        int batch_size,
        int num_heads)
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16, batch_size * num_heads);

        QKmatmulKernel<<<grid, block>>>(Q, Kt, out, M, N, num_heads);

        cudaDeviceSynchronize();
    }

    void softmax2D(
        float *arr, // (B, T, vocab_size)
        float *out,
        int batch_size,
        int seq_len,
        int vocab_size)
    {

        dim3 grid(seq_len, batch_size);
        dim3 block(min(((vocab_size + 31) / 32) * 32, 1024));

        SoftmaxKernel2D<<<grid, block>>>(arr, out, batch_size, seq_len, vocab_size);

        cudaDeviceSynchronize();
    }

    void softmax(
        float *arr, //  Shape(batch_size, n_head, T, T)
        float *out,
        int N,
        int seq_len,
        int n_head,
        int batch_size)
    {
        // Here if the T dimension is less than 32 then we can talk with the
        // internel registers in our thread and softmaxKernel4D will be enough.
        // else we will need to use other kernel

        // Note:- this probally will not scale.

        int rows = batch_size * n_head * seq_len; // each row gets softmaxed, and this 2D works for 2D

        if (seq_len > 32)
        {

            dim3 grid(rows, 1);
            dim3 block(min(((seq_len + 31) / 32) * 32, 1024));

            SoftmaxKernel2D<<<grid, block>>>(arr, out, batch_size, rows, seq_len);
        }
        else
        {
            dim3 grid(seq_len, n_head, N);
            dim3 block(min(((seq_len + 31) / 32) * 32, 1024));
            softmaxKrenel4D<<<grid, block>>>(arr, out, N, seq_len, n_head);
        }
        cudaDeviceSynchronize();
    }

    void ScalerDvisionElem(
        float *arr, // (batch, n_head, seq_len, seq_len)
        int batch,
        int n_head,
        int seq_len,
        int head_dim)
    {
        int total = batch * n_head * seq_len * seq_len;
        float scale = sqrtf((float)head_dim);

        dim3 grid(seq_len, n_head, batch);
        dim3 block(seq_len);

        ScalerDvisionDModelKernel<<<grid, block>>>(arr, total, scale, n_head, seq_len);

        cudaDeviceSynchronize();
    }
    void vectorKernel(
        float *A,
        float *B,
        float *C,
        int N)
    {
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;

        vectorAddKernel<<<numBlocks, blockSize>>>(A, B, C, N);
        // this might look confusing but C is the output.
        cudaDeviceSynchronize();
    }

    // What we expect from this method do do is
    // swap dimension like
    /*
      Before:- Shape(batch_size, T, n_head, d_head)
      After:- Shape(batch_size, n_head, T, d_head)
    */
    void SwapNS(
        int num_heads,
        int head_dim,
        float *arr,     // Shape(batch_size, num_heads, seq_len, head_dim)
        float *out,     // Shape(batch_size,seq_len, num_head, head_dim)
        int batch_size, // batch_size
        int seq_len,    // seq_len
        bool reverse)
    {
        dim3 block(head_dim);
        dim3 grid(batch_size * seq_len, num_heads);

        TransposeKernel<<<grid, block>>>(num_heads, head_dim, arr, out, batch_size, seq_len, reverse);

        cudaDeviceSynchronize();
    }
}