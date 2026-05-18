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
    // each id is pinned to a thread itself.
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int pos = blockDim.y * blockIdx.y + threadIdx.y; // because thread runs in parallel we use this formula to look our positions in a gird

    /*
        Note:- Okay so inorder to solve something like this, first we need to change the prespection on how we approach the problem
        because we are using differnet hardware to perform a given task. Although, this task is easily done by the CPU, just to get momentum in this
        platofm we will be using a GPU to use positional encoding using a GPU.
    */
    if (pos < seq_len && id < d_model)
    {
        float denominator = powf(10000.0f, (2.0f * (id / 2)) / d_model);
        int idx = pos * d_model + id; // Flattened 1D array index

        if (id % 2 == 0)
        {
            out[idx] = sinf(pos / denominator);
        }
        else
        {
            out[idx] = cosf(pos / denominator);
        }
    }
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
        const int threads_per_block = 256;
        const int blocks_per_grid = (seq_len + threads_per_block - 1) / threads_per_block;

        positional_embedding_kernel<<<blocks_per_grid, threads_per_block>>>(out, seq_len, d_model);

        cudaDeviceSynchronize();
    }
}
