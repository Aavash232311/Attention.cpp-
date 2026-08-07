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

// come on youre smart you will make it.
__global__ void LookUpKernel(
    int *x,            // Shape(seq_len, batch_size)
    float *embeddings, // Shape(vocab_size, d_mdoel)
    float *C,          // B, T, C
    int d_model,
    int seq_len,
    int batch_size,
    int vocab_size)
{
    // We want the loopup into embeddings table.
    int rows = blockIdx.x; // this is future me reading god you were a genious.
    int cols = blockIdx.y;
    int e = threadIdx.x;

    if (rows >= seq_len || cols >= batch_size)
        return;

    int index = (rows * batch_size) + cols;
    // for each entry of x, you grab matching row,
    int valX = x[index]; // now this value will be used to search on lookup table.

    // (row * size) + cols, remember 8-5 react dev the formula you used when debugging the kernel
    int indexB = (valX * d_model) + e; // when the kenrnel launches its distrubuted acrosss the matrix B in our case in embeddins so we are accounting for threads.
    float valB = embeddings[indexB];

    // Final shape of the output we would want it to be Shape(seq_len, batch_size, d_model)
    int out_idx = cols * (seq_len * d_model) + rows * d_model + e;
    C[out_idx] = valB;
    // if we were do another kenel launch then data gets transfered from PCIe express BUS that is costly lets so that here.
    // Lets think through here.
    // Things are happenning in parallel and we could end up writing one memory address many times so
    // Leave it we will do in different kernel launch.
    // if we were to implement adding the embeddings here as well then it would be known as kernel fusion.
}

// We need more and more and more kernel fussion here since I am learning I did it in my way.
// writing in GPU is hard and understanding takes a time so. we can refine it later :)

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


__global__ void FinalEmbeddingKernel(
    float *lookedUpEmbeddings, // Shape BTC
    float *sinosudialEncoding, // Shape (seq_len, d_model)
    float *C,                  // Shape (batch_size, seq_len, d_model)  BTC
    int d_model,
    int seq_len,
    int batch_size)
{
    int rows = blockIdx.x; // seq position (s)
    int cols = blockIdx.y; // batch index (b)
    int e = threadIdx.x;

    if (rows >= seq_len || cols >= batch_size || e >= d_model)
        return;

    int idxSinosudialEncoding = rows * d_model + e; // (seq_len, d_model) — unaffected

    // output (matches declared C shape)
    int idx = cols * (seq_len * d_model) + rows * d_model + e;
    C[idx] = lookedUpEmbeddings[idx] + sinosudialEncoding[idxSinosudialEncoding];
}


extern "C"
{
    void positionalEmbeddings(float *out, int seq_len, int d_model)
    {
        dim3 block(16, 16); // 16x16=256 threads in total.

        dim3 grid(
            (d_model + 15) / 16, // this celling division determines how many of those 16x16 grid are needed to cover entire dimension.
            (seq_len + 15) / 16);

        positional_embedding_kernel<<<grid, block>>>(out, seq_len, d_model);

        cudaDeviceSynchronize();
    }

    void lookup(
        int *x,            // something like seq x batch_size
        float *embeddings, // te (vocab_size * d_model)
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

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
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
}