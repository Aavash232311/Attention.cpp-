#include <stdio.h>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <ranges>
#include <fstream>
#include <iomanip>

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

int main()
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