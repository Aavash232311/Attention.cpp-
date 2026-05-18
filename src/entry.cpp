#include "include/helper.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <cstdio>
// nvcc src/entry.cpp src/kernel/math.cu -o src/bin/entry  ./src/bin/entry
extern "C" void softmax(float *arr, float *out, int N);
extern "C" void postionalEmbeddings(float *dimenstion, int N);

// please don't judge me this is my first semester learning this particular language with little background in C.
// I will make this messy in order to learn

int main()
{
    int N = 3;
    float arr[N] = {2.0f, 1.0f, 0.0f}; // this is in the Random Access Memory in the stack
    size_t size = sizeof(arr);
    // lets try to understand everything here
    float *host_a = (float *)malloc(size);
    float *host_out = (float *)malloc(size);

    // copy to memory allocated in the heap

    for (int i = 0; i < N; i++)
    {
        host_a[i] = arr[i];
    }

    // now allocate memory on the GPU
    float *device_arr_a, *device_arr_out;
    cudaMalloc((void **)&device_arr_a, size);
    cudaMalloc((void **)&device_arr_out, size);

    // copy input to device
    cudaMemcpy(device_arr_a, host_a, size, cudaMemcpyHostToDevice);

    // launch kernel
    softmax(device_arr_a, device_arr_out, N);
    cudaDeviceSynchronize(); 

    // copy output back to host
    cudaMemcpy(host_out, device_arr_out, size, cudaMemcpyDeviceToHost);
    cudaFree(device_arr_a);
    cudaFree(device_arr_out);
    device_arr_a = nullptr;
    device_arr_out = nullptr;


    free(host_a);
    free(host_out);
    host_a = nullptr;
    host_out = nullptr;
    // Lets work with positional embeddings
    // In modern day torch it is handelled by encoding.

    Helper helper;
    std::string input_test = "Hello world";

    int input_length = input_test.length();

    auto encoded_input = helper.encoder("Hello world");
    // helper.showHashMap(encoded_input, input_length);

    helper.decoder(encoded_input, input_length);

    // we intentionally want to allocate memory in heap here.
    // It's doing to be in Rnadom Access Memory for a while
    // This input maybe be something large, and when after encoding we will delete it.

    auto textEncoderFile = std::make_unique<EncoderText>();
    // that's cool no delete required
    textEncoderFile->loadTextChunk("./src/data/chunk.txt");
}