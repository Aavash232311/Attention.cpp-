#include <iostream>
#include <iterator>
#include <math.h>
#include <mma.h>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

// This is sinosudial positional embeddings. For simpilicity we wont used learned positional embeddings. We can learn about it later atleast.
__global__ void positional_embedding_kernel(float *out, int seq_len, int d_model)
{
    // we need to change how we think here because we are working in low level and the things are happenning in the parallel.

    int pos = blockIdx.y * blockDim.y + threadIdx.y; // this is row
    int k = blockIdx.x * blockDim.x + threadIdx.x;   // this is column

    if (pos >= seq_len || k >= d_model)
        return;

    // Lets understand the shape of this positional embddings (d_model, seq_len)

    int i = k / 2;
    float denom = powf(10000.0f, 2.0f * i / d_model);

    float sin_val, cos_val;
    sincosf(pos / denom, &sin_val, &cos_val); // to prevent the branch divergence
    out[pos * d_model + k] = (k % 2 == 0) ? sin_val : cos_val;
}

__global__ void addVec(const float *a, const float *b, float *c, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

// come on youre smart you will make it.
__global__ void LookUpKernel(
    int *x,            // Shape(seq_len, batch_size)
    float *embeddings, // Shape(vocab_size, d_mdoel)
    float *C,
    int d_model,
    int seq_len,
    int batch_size,
    int vocab_size)
{
    // We want the loopup into embeddings table.
    int rows = blockIdx.x;
    int cols = blockIdx.y;
    int e = threadIdx.x;

    if (rows >= seq_len || cols >= batch_size)
        return;

    int index = (rows * batch_size) + cols;
    int valX = x[index]; // now this value will be used to search on lookup table.

    int indexB = (valX * d_model) + e; // when the kenrnel launches its distrubuted acrosss the matrix B in our case in embeddins so we are accounting for threads.
    float valB = embeddings[indexB];

    // Final shape of the output we would want it to be Shape(seq_len, batch_size, d_model)
    C[rows * batch_size * d_model + cols * d_model + e] = valB;
    // if we were do another kenel launch then data gets transfered from PCIe express BUS that is costly lets so that here.
    // Lets think through here.
    // Things are happenning in parallel and we could end up writing one memory address many times so
    // Leave it we will do in different kernel launch.
    // if we were to implement adding the embeddings here as well then it would be known as kernel fusion.
}

__global__ void FinalEmbeddingKernel(
    float *lookedUpEmbeddings, // Shape (batch_size, seq_len, d_model)
    float *sinosudialEncoding, // Shape (seq_len, d_model)
    float *C,                  // Shape(batch_size, seq_len, d_model)
    int d_model,
    int seq_len,
    int batch_size)
{
    int rows = blockIdx.x; // This gives the row and col of grid.
    int cols = blockIdx.y;
    int e = threadIdx.x;

    if (rows >= seq_len || cols >= batch_size)
        return;

    int idxSinosudialEncoding = (rows * d_model) + e; // width = d_model of sinosudial encoding

    int idxLookedUpEmbeddings = rows * batch_size * d_model + cols * d_model + e;

    C[idxLookedUpEmbeddings] = lookedUpEmbeddings[idxLookedUpEmbeddings] + sinosudialEncoding[idxSinosudialEncoding];
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

    c[(rows * N) + cols] = sum + b[rows];
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

__global__ void multiHeadedAttentionKernel(
    int num_head,
    int head_dimension,
    float *ws,
    float *out,
    int M,
    int N,
    int K)
{
    int token_idx = blockIdx.x; // which token (0 to M*N)
    int head_idx = blockIdx.y;  // which head
    int hd_idx = threadIdx.x;   // which elelemnt

    // idx = (rows * width) + cols
    int idx = token_idx * (num_head * head_dimension) + head_idx * (head_dimension) + hd_idx;
    out[idx] = ws[idx];
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
__global__ void TransposeKernel(
    int num_heads,
    int head_dim,
    float *arr, // Shape(batch_size, seq_len, n_head, d_head)
    float *out, // Shape(batch_size, n_head, seq_len, d_model)
    int M,      // batch_size
    int N,      // d_model
    int K,
    bool reverse = true) // seq_len
{
    int rows = blockIdx.x; // we are launching in such a way that block(seq_len, n_head)
    int cols = blockIdx.y;

    int hd_idx = threadIdx.x; // current element idx

    int batch_idx = rows / K; // K = seq_len
    int seq_idx = rows % K;   // K = seq_len
    // and each thread inside of this has d_head shpae for example 64
    // our grid is (batch_size x seq_len, n_head, head_dim)
    int idx_curr = rows * (num_heads * head_dim) + cols * head_dim + hd_idx;

    // Shape(batch_size, n_head, seq_len, d_head)
    int out_idx = batch_idx * (num_heads * K * head_dim) + cols * (K * head_dim) + seq_idx * head_dim + hd_idx;
    if (reverse)
        out[idx_curr] = arr[out_idx];
    else
        out[out_idx] = arr[idx_curr];
}

__global__ void TransposeKeyKernel(
    int num_heads,
    int head_dim,
    float *arr, // Shape (batch_size, n_head, seq_len, head_dim)
    float *out, // Shape (batch_size, n_head, head_dim, seq_len)
    int M,      // batch_size,
    int N,      // d_head
    int K,      // seq_len
    bool reverse = true)
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

// Only the last two batch are multiplied here rest of them are taken along
__global__ void QKmatmulKernel(
    float *Q,  // Shape(batch_size, n_head, seq_len, head_dim)
    float *Kt, // Shape(batch_size, n_head, head_dim, seq_len)
    float *out, // (batch, n_head, seq_len, seq_len)
    int M, // (seq_len, head_dim) = (M, N)
    int N, // (head_dim, seq_len) = (N, M)
    int n_head)
{
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;

    if (rows >= M || cols >= N)
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

extern "C"
{
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
    void SwapNS(
        int num_heads,
        int head_dim,
        float *arr, //  Shape(batch_size, seq_len, n_head, d_head)
        float *out, // Shape(batch_size, n_head, seq_len, d_head)
        int M,      // batch_size
        int N,      // d_head
        int K,      // seq_len
        bool reverse)
    {
        dim3 block(head_dim);
        dim3 grid(M * K, num_heads);

        TransposeKernel<<<grid, block>>>(num_heads, head_dim, arr, out, M, N, K, reverse);

        cudaDeviceSynchronize();
    }

    void multiHeadedAttention(
        int num_head,
        int head_dimension,
        float *ws,
        float *out, //  B, T, C, n_head, head_dim
        int M,      // batch_size
        int K,      // d_model
        int N)      // seq_len, say we have n_head=8, head_dim=64, d_model=512
    {
        dim3 block(head_dimension);
        dim3 grid(M * N, num_head); // MXN will ignore btach_size, seq_len

        multiHeadedAttentionKernel<<<grid, block>>>(num_head, head_dimension, ws, out, M, K, N);

        cudaDeviceSynchronize();
    }

    void WeightedSum(
        float *x, // Shape(M, K)
        float *w, // Shape(K, N)
        float *b, // Shape(M, N)
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
    void addEmbeddings(
        float *lookedUpEmbeddings,
        float *sinosudialEncoding,
        float *C,
        int d_model,
        int seq_len,
        int batch_size)
    {
        dim3 gird(seq_len, batch_size);
        int threads = min(d_model, 1024);
        dim3 block(threads);

        FinalEmbeddingKernel<<<gird, block>>>(lookedUpEmbeddings, sinosudialEncoding, C, d_model, seq_len, batch_size);

        cudaDeviceSynchronize();
    }
    void lookup(
        int *x,
        float *embeddings,
        float *C,
        int d_model,
        int seq_len,
        int batch_size,
        int vocab_size)
    {
        // We are launching it such that each element in the x maps to every element in the embeddings
        // Its definately going to take some time for me to derive the problem.
        // Util and unless I am not able to derive things its not the actual problem solving.
        dim3 grid(seq_len, batch_size);
        int threads = min(d_model, 1024);
        dim3 block(threads); // we have a hardware limit here so.

        // whever you get confused think of A[0] as 0 lets say that needs to be mapped d_moel times in order to write the rows.

        LookUpKernel<<<grid, block>>>(x, embeddings, C, d_model, seq_len, batch_size, vocab_size);

        cudaDeviceSynchronize();
    }
    void softmax(float *arr, float *out, int N)
    {

        const int threads_per_block = 256;
        const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

        softmax_kernel<<<blocks_per_grid, threads_per_block>>>(arr, out, N);

        // wait for GPU to finish so the print statements display
        cudaDeviceSynchronize();
    }
    void positionalEmbeddings(float *out, int seq_len, int d_model)
    {
        dim3 block(16, 16); // 16x16=256 threads in total.

        dim3 grid(
            (d_model + 15) / 16, // this celling division determines how many of those 16x16 grid are needed to cover entire dimension.
            (seq_len + 15) / 16);

        positional_embedding_kernel<<<grid, block>>>(out, seq_len, d_model);

        cudaDeviceSynchronize();
    }
}
