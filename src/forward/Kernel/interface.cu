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
}