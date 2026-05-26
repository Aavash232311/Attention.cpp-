#include "include/helper.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <cstdio>

// This is for unit testing features and compoenents.
// We will also create a cheat sheet in python to compare and check if something is off.
// Working with a different hardware is really tuff.


// I will use this to check if all the components are working.
// nvcc src/entry.cpp src/kernel/math.cu -o src/bin/entry  ./src/bin/entry
extern "C" void softmax(float *arr, float *out, int N);
extern "C" void positionalEmbeddings(float *out, int seq_len, int d_model);

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

    auto textEncoderFile = std::make_unique<EncoderText>();
    // that's cool no delete required
    std::string filePath = "./src/data/chunk.txt";
    textEncoderFile->loadTextChunk(filePath);

    auto& charPool = textEncoderFile->getFileAsChar(); // getting this as a ref.

    auto helper = std::make_unique<Helper>(charPool);
    std::string input_text = "RelocateAvailable";
    auto encodedMap = helper->encoder(input_text);

    auto decoded = helper->decoder(encodedMap);


    // positional encoding like the original attention paper, not learned.
    int d_model = 8;
    int seq_len = input_text.length(); // since we are working on character level model.

    int total_shape_positional_encoding = d_model * seq_len;
    float *psoitional_encoding_out = (float *)malloc(total_shape_positional_encoding * sizeof(float));

    float *device_positional_encoding;
    cudaMalloc((void **)&device_positional_encoding, total_shape_positional_encoding * sizeof(float));

    positionalEmbeddings(device_positional_encoding, seq_len, d_model);

    cudaMemcpy(psoitional_encoding_out, device_positional_encoding, total_shape_positional_encoding * sizeof(float), cudaMemcpyDeviceToHost);

   
    cudaFree(device_positional_encoding);
    device_positional_encoding = nullptr;
    free(psoitional_encoding_out);
    psoitional_encoding_out = nullptr;
}