#include <stdio.h>
#include <random>
#include <cuda_runtime.h>


__global__ void addVec(const float *a, const float *b, float *c, int N) {
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < N) {
     c[i] = a[i] + b[i];
   }
}


int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);

    std::mt19937 gen; 

    float min = 1.0f;
    float max = 5.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::uniform_real_distribution<float> dis(min, max);

    // Allocate memeory in CPU
    float *host_arr_a = (float *)malloc(size);
    float *host_arr_b = (float *)malloc(size);
    float *host_arr_c = (float *)malloc(size);

    // random values in a vector
    for (int i = 0; i < N; i++) {
        host_arr_a[i] = dis(gen);
        host_arr_b[i] = dis(gen);
    }

    // Allocating memory in CPU
    float *device_arr_A, *device_arr_B, *device_arr_C;
    cudaMalloc((void **)&device_arr_A, size);
    cudaMalloc((void **)&device_arr_B, size);
    cudaMalloc((void **)&device_arr_C, size);

    cudaMemcpy(host_arr_a, device_arr_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(host_arr_b, device_arr_B, size, cudaMemcpyHostToDevice);

    // kernel launch
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    cudaEventRecord(start);
    addVec<<<blocks_per_grid, threads_per_block>>>(device_arr_A, device_arr_B, device_arr_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop); // we need to wait for the GPU to stop before finishing
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(host_arr_c, device_arr_C, size, cudaMemcpyHostToDevice);

    cudaFree(device_arr_A);
    cudaFree(device_arr_B);
    cudaFree(device_arr_C);

    free(host_arr_a);
    free(host_arr_b);
    free(host_arr_c);
}