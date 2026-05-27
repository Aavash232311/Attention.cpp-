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
    int row = blockIdx.x;
    int col = blockIdx.y;

    int threadIx = threadIdx.x;
    printf("Rows: %d Cols: %d ThreadIx: %d \n", row, col, threadIx);
}

int main()
{
    float A[9] = {0};
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

    cudaMemcpy(A, device_arr_A, 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B, device_arr_B, 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C, device_arr_C, 8 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(3, 3); // here meaning we have 81 threasds in total.
    dim3 block(9); // this means we have 9 threads per block.

    kernel<<<grid, block>>>(device_arr_A, device_arr_B, device_arr_C);

    cudaMemcpy(device_arr_C, C, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Hello world! " << endl;

    cudaFree(device_arr_A);
    cudaFree(device_arr_B);
    cudaFree(device_arr_C);
}