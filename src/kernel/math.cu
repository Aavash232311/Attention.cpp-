#include <iostream>
#include <iterator>
#include <math.h>
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
    int rows = blockIdx.x;
    int cols = blockIdx.y;

    int idx = cols * x + rows;
    if (rows >= y || cols >= x)
        return;

    // registers are ultra fast memory within SM's in the GPU
    curandState local = state[idx]; // global mem to register ex pos 0
    float std = sqrtf(2.0f / (float)y);
    arr[idx] = curand_normal(&local) * std; // local goes to pos 1

    // That opreation from global memory to register is physcially copied
    // back and fourh between the register and global memory
    state[idx] = local; // register -> global position 1, we need to write back the global because it is the only memory that survives after the Kernel ends.
    // if the each thread have different position on the global memory then why do we care about writing it back?
    // well the answer is if we using this KaimingInit again and the data is physically
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
        int max_threads = 1;
        dim3 grid(x, y);
        dim3 block(max_threads);

        SetUpRnd<<<x * y, max_threads>>>(state, seed, x * y);
        KaimingInitKernel<<<grid, block>>>(arr, state, x, y);

        cudaDeviceSynchronize();
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
