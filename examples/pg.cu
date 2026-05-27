#include <stdio.h>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <ranges>
#include <fstream>
#include <iomanip>

using namespace std;

__global__ void kernel(float *A, float *B, float *C)
{
    int rows = blockIdx.y * blockDim.y + threadIdx.y; // rows
    int cols = blockIdx.x * blockDim.x + threadIdx.x; // cols

    if (!(rows < 3 &&  cols < 3)) return;

    int index = (rows * 3) + cols;
    float val = A[index];

    int threadIx = threadIdx.x;

    printf("Rows: %d Cols: %d ThreadIx: %d value %.1f \n", rows, cols, threadIx, val);
}

int main()
{
    float A[9] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8};

    float B[9] = {0};
    float C[9];

    /*
    We can imagine this float A matrix as something like this

    A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    } smilialry for B

    */

    float *device_arr_A, *device_arr_B, *device_arr_C;
    cudaMalloc((void **)&device_arr_A, 9 * sizeof(float));
    cudaMalloc((void **)&device_arr_B, 9 * sizeof(float));
    cudaMalloc((void **)&device_arr_C, 9 * sizeof(float));

    cudaMemcpy(device_arr_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_arr_B, B, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_arr_C, C, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(3, 3);
    dim3 block(2);

    kernel<<<grid, block>>>(device_arr_A, device_arr_B, device_arr_C);

    cudaDeviceSynchronize();
    cudaMemcpy(device_arr_C, C, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_arr_A);
    cudaFree(device_arr_B);
    cudaFree(device_arr_C);
}