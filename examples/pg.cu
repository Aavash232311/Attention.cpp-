#include <stdio.h>
#include "../src/include/helper.h"
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <ranges>
#include <fstream>
#include <iomanip>

// THIS IS MY ROUGH WORK

// Just ignore this mesh this is for me to unit test, becase
// a trasnformer running in training on batch epoch and flying in heven when using
// parallel computing hard to debug so I will test the thing here.
// test the equivalent code in python and then add the feature.

using namespace std;

__global__ void kernel(int *A, float *B, float *C, int embedding)
{
    int rows = blockIdx.y;
    int cols = blockIdx.x;
    int e = threadIdx.x;

    int index = (rows * 3) + cols;
    int val = A[index];

    int indexB = (val * 6) + e;
    float valB = B[indexB];

    C[rows * gridDim.x * embedding + cols * embedding + e] = valB;
    /*

    rows * gridDim.x * embeddings
    skip row, each one of them has 3 position and final embeddings have 15 values.

    + how for how many cols to skip cols * embeddings + threadId

    */
}

__global__ void positional_embedding_kernel(float *out, int seq_len, int d_model)
{
    int pos = blockIdx.y * blockDim.y + threadIdx.y; // row
    int k = blockIdx.x * blockDim.x + threadIdx.x;   // col

    if (pos >= seq_len || k >= d_model)
        return;

    int i = k / 2;
    float denom = powf(10000.0f, 2.0f * i / d_model);

    float sin_val, cos_val;
    sincosf(pos / denom, &sin_val, &cos_val);
    out[pos * d_model + k] = (k % 2 == 0) ? sin_val : cos_val;
}

unique_ptr<Utility> utils = make_unique<Utility>();

void loopUpTest()
{
    int A[9] = {
        0, 1, 2,
        2, 1, 0,
        1, 2, 1};

    float B[36] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
        19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
        31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f};

    /*
    We can imagine this float A matrix as something like this

    A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    } smilialry for B

    */

    int *device_arr_A;
    float *device_arr_B, *device_arr_C;
    cudaMalloc((void **)&device_arr_A, 9 * sizeof(int));
    cudaMalloc((void **)&device_arr_B, 36 * sizeof(float));
    cudaMalloc((void **)&device_arr_C, 3 * 3 * 6 * sizeof(float));

    cudaMemcpy(device_arr_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_arr_B, B, 36 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(3, 3);
    dim3 block(6);
    kernel<<<grid, block>>>(device_arr_A, device_arr_B, device_arr_C, 6);

    cudaDeviceSynchronize();
    float C[3 * 3 * 6];
    cudaMemcpy(C, device_arr_C, 3 * 3 * 6 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_arr_A);
    cudaFree(device_arr_B);
    cudaFree(device_arr_C);

    for (int r = 0; r < 3; r++)
    {
        printf("t%d:\n", r);
        for (int c = 0; c < 3; c++)
        {
            printf("  col%d: [", c);
            for (int e = 0; e < 3; e++)
            {
                printf("%.1f ", C[r * 3 * 6 + c * 6 + e]);
            }
            printf("]\n");
        }
    }
}

void positionalEncodingTest()
{
    int d_model = 8;
    int seq_len = 4;
    int shape = d_model * seq_len;
    float *positionalEncodingOut = (float *)malloc(shape * sizeof(float)); // host memeory

    float *devicePositionalEncoding;
    cudaMalloc((void **)&devicePositionalEncoding, shape * sizeof(float)); // GPU memory

    dim3 block(16, 16); // 16x16=256 threads in total.

    dim3 grid(
        (d_model + 15) / 16, // this celling division determines how many of those 16x16 grid are needed to cover entire dimension.
        (seq_len + 15) / 16);

    positional_embedding_kernel<<<grid, block>>>(devicePositionalEncoding, seq_len, d_model);

    cudaDeviceSynchronize();

    cudaMemcpy(positionalEncodingOut, // copy back
               devicePositionalEncoding,
               shape * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(devicePositionalEncoding);

    free(positionalEncodingOut); // in the attention.cpp this is free in distructor which is fine since it is created in constructor
}

/*
Lets keep track of the picture that we might have here.

W = [[1,  2],     x = [[3],    b = [[1],
     [-1, 3],          [1]]         [-2],
     [4,  0]]                       [0]]

z₁ = (1×3) + (2×1) + 1 = 3 + 2 + 1 = 6
*/

/*

w = [1, 2, -1, 3, 4, 0]

x = [3, 1]

B(1, N)
W (M, K); X(k, N) Z(M,N) = output

M = 2; K = 2
K = 2, N = 1

COMMON should be K in order for this to be a valid matrix multiplication

At Position(0, 0)
at l = 0;
x[0 * 2 + 0] = 1
w[0 * 1 + 0] = 3

at l = 1
x[0 * 2 + 1] = 2
w[1 * 1 + 0] = 1

sum = 5;


// You may not unfload this code logic at once but if you think through then you can. Its not that hard.

*/

// __global__ void LinearTransformation(
//     float *x,
//     float *w,
//     float *b,
//     float *c,
//     int M,
//     int K,
//     int N)
// {
//     /*
//         W (M, K); X(k, N)

//         M = 2; K = 2
//         K = 2, N = 1
//     */

//     int rows = blockIdx.y * blockDim.y + threadIdx.y; // row
//     int cols = blockIdx.x * blockDim.x + threadIdx.x; // col

//     if (rows >= M || cols >= N)
//         return;

//     // formula for idx = (rows * width) + cols
//     float sum = 0.0f;
//     for (int row_b = 0; row_b < K; ++row_b)
//     {

//         float valA = x[(rows * K) + row_b];
//         float valB = w[(row_b * N) + cols];
//         sum += (valA * valB);
//     }

//     c[(rows * N) + cols] = sum + b[cols];
// }

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

void LinearTransformationTest()
{
    float W[12] = {
        2, 0, -1,
        1, 3, 2,
        -2, 1, 4,
        0, -1, 3};

    float X[9] = {
        4, 1, 0,
        -1, 2, 3,
        2, 5, -2};

    float b[4] = {1, -3, 2, 0};
    float res[4 * 3];

    float *device_a;
    float *device_b;
    float *device_c;
    float *device_out;

    cudaMalloc((void **)&device_a, 4 * 3 * sizeof(float));
    cudaMalloc((void **)&device_b, 3 * 3 * sizeof(float));
    cudaMalloc((void **)&device_c, 4 * sizeof(float));
    cudaMalloc((void **)&device_out, 4 * 3 * sizeof(float));

    cudaMemcpy(device_a, W, 4 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, X, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, b, 4 * sizeof(float), cudaMemcpyHostToDevice);


    int M = 4;
    int N = 3;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,  
              (M + block.y - 1) / block.y); 

    WeightedSumKernel<<<grid, block>>>(
        device_a,
        device_b,
        device_c,
        device_out,
        4, // M
        3, // K
        3  // N
    );
    cudaDeviceSynchronize();

    cudaMemcpy(res, device_out, 4 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaFree(device_out);

    utils->printFlatArray2D(res, 4, 3);
}

int main()
{
    // loopUpTest();
    // positionalEncodingTest();

    LinearTransformationTest();
}