#include <iostream>
#include <iterator>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>

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
    int k = blockIdx.x * blockDim.x + threadIdx.x; // this is column

    if (pos >= seq_len || k >= d_model)
        return;

    int i = k / 2;
    float denom = powf(10000.0f, 2 * i / d_model);

    float sin_val, cos_val;
    sincosf(pos / denom, &sin_val, &cos_val); // to prevent the branch divergence
    out[pos * d_model + k] = (k % 2 == 0) ? sin_val : cos_val;
}

__global__ void addVec(const float *a, const float *b, float *c, int N) {
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < N) {
     c[i] = a[i] + b[i];
   }
}


// For adding up embeddings
__global__ void addEmbeddingsKernel()
{
    
}


extern "C"
{
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
            (seq_len + 15) / 16  
        );

        positional_embedding_kernel<<<grid, block>>>(out, seq_len, d_model);

        cudaDeviceSynchronize();
    }
}
