#pragma once
#include "../include/helper.hpp"
#include "../include/utils.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

extern "C" void KaimingInit(float *arr, curandState *state, int x, int y, unsigned long seed);
extern "C" void WeightedSum(float *x, float *w, float *b, float *c, int M, int K, int N);

// in math.cu kenrel -Reused logic
extern "C" void multiHeadedAttention(int num_head, int head_dimension, float *ws, float *out, int batch_size, int d_model, int seq_len);
extern "C" void SwapNS(int num_head, int head_dimension, float *ws, float *out, int batch_size, int seq_len, bool reverse);
extern "C" void TransposeKey(float *arr, float *out, int num_heads, int head_dim,  int batch_size, int seq_len, bool reverse);

class Linear
{
    bool dLinear = true;
    float *ws = nullptr; // weighted sum
    float *weight = nullptr;
    float *bias = nullptr;
    float *x = nullptr;

    float *h_mha;

    float *d_x;
    float *d_w;
    float *d_b;
    float *d_out;

    int f_in;
    int f_out;
    int seq_len;
    int batch_size;
    std::unique_ptr<Utility> utils = std::make_unique<Utility>();

    float *mhead_out_host = nullptr; // output after multi head attention
    float *mhead_out_device;         // this is multi head attention out device
    float *device_hhead_in;          // copy to this from weighted sum

    float n_head;
    float head_dim;

    bool debug = true;

    float *deviceArrInTranspose;

public:
    Linear(int feature_in, int feature_out, int seq_len, int batch_size, int n_head, bool debug)
    {
        this->seq_len = seq_len;
        this->batch_size = batch_size;
        this->f_in = feature_in;
        this->f_out = feature_out;
        this->n_head = n_head;
        this->head_dim = feature_out / n_head;
        this->LinearParams(feature_in, feature_out);
        this->debug = debug;

        // pre-allocate memroy in the constructor
        // because we need to flattern this, x (batch_size, seq_len, d_model)

        this->ws = (float *)malloc(seq_len * batch_size * f_out * sizeof(float));

        cudaMalloc((void **)&d_x, seq_len * batch_size * f_in * sizeof(float));
        cudaMalloc((void **)&d_w, f_in * f_out * sizeof(float)); // (M, K)
        cudaMalloc((void **)&d_b, f_out * sizeof(float));
        cudaMalloc((void **)&d_out, seq_len * batch_size * f_out * sizeof(float));

        // copy to the device, those allocated weight and biases.
        cudaMemcpy(d_w, weight, f_in * f_out * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, bias, f_out * sizeof(float), cudaMemcpyHostToDevice);

        /// allocate the memory for multi headed attention Shape(B, T, num_heads, head_dim), JUST BTC when in
        cudaMalloc((void **)&device_hhead_in, seq_len * batch_size * f_out * sizeof(float)); // copy ws here

        this->mhead_out_host = (float *)malloc(seq_len * batch_size * n_head * head_dim * sizeof(float));

        cudaMalloc((void **)&mhead_out_device, seq_len * batch_size * n_head * head_dim * sizeof(float));

        // In order to transpose that Q,K,V and swap we need to allocate the memory in GDDR VRAM
        cudaMalloc((void **)&deviceArrInTranspose, batch_size * seq_len * n_head * head_dim * sizeof(float));
    }

    ~Linear()
    {
        (ws != nullptr ? free(ws) : void());
        (weight != nullptr ? free(weight) : void());
        (bias != nullptr ? free(bias) : void());

        (mhead_out_host != nullptr ? free(mhead_out_host) : void());

        cudaFree(d_x);
        cudaFree(d_w);
        cudaFree(d_b);
        cudaFree(d_out);

        cudaFree(deviceArrInTranspose);

        cudaFree(mhead_out_device);
    }

    void LinearParams(int fan_in, int fan_out)
    {
        weight = (float *)malloc(fan_in * fan_out * sizeof(float)); // these are out for both weight and biases.
        bias = (float *)malloc(fan_out * sizeof(float));

        float *device_weight;
        float *device_bias;
        curandState *d_state_weight;
        curandState *d_state_bias;

        cudaMalloc((void **)&device_weight, fan_in * fan_out * sizeof(float));
        cudaMalloc(&d_state_weight, fan_in * fan_out * sizeof(curandState));
        cudaMalloc(&d_state_bias, fan_out * sizeof(curandState));
        cudaMalloc((void **)&device_bias, fan_out * sizeof(float));

        // we need tow kernal launches here
        KaimingInit(device_weight, d_state_weight, fan_in, fan_out, 42);
        KaimingInit(device_bias, d_state_bias, 1, fan_out, 43);

        cudaMemcpy(weight, device_weight, fan_in * fan_out * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bias, device_bias, fan_out * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(device_weight);
        cudaFree(device_bias);
        cudaFree(d_state_weight);
        cudaFree(d_state_bias);

        if (!(debug))
        {
            std::cout << "Weight" << std::endl;
            utils->printFlatArray2D(weight, fan_in, fan_out);
            std::cout << "Bias" << std::endl;
            utils->printFlatArray2D(this->bias, fan_out, 1);
            std::cout << "Fan in: " << fan_in << " Fan out: " << fan_out << std::endl;
        }
    }

    float *getBias()
    {
        return this->bias;
    }

    float *getWeight()
    {
        return this->weight;
    }

    float *forward(float *val)
    {
        this->x = val;
        int m = batch_size * seq_len;

        cudaMemcpy(d_x, x, seq_len * batch_size * f_in * sizeof(float), cudaMemcpyHostToDevice);

        WeightedSum(d_x, d_w, d_b, d_out, m, f_in, f_out);

        cudaMemcpy(ws, d_out, seq_len * batch_size * f_out * sizeof(float), cudaMemcpyDeviceToHost);

        if (debug)
        {
            // std::cout << "Weight" << std::endl;
            // utils->printFlatArray2D(weight, f_in, f_out);

            // std::cout << "Bias" << std::endl;
            // utils->printFlatArray3D(bias, 1, 1, f_out);

            // utils->printFlatArray3D(ws, batch_size, seq_len, f_out, true);
        }

        dLinear = false;

        return ws;
    }

    float *reshapeHead() // make it have more dim so that effective computation can happen in parallel.
    {
        // copy that weighted sum into device so we can split it down.
        cudaMemcpy(device_hhead_in, ws, seq_len * batch_size * f_out * sizeof(float), cudaMemcpyHostToDevice);

        multiHeadedAttention(n_head, head_dim, device_hhead_in, mhead_out_device, batch_size, f_out, seq_len);

        cudaMemcpy(mhead_out_host, mhead_out_device, seq_len * batch_size * n_head * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

        // Weighted sum looks healthy here.
        if (dLinear)
        {
            // utils->printFlatArray3D(ws, seq_len, batch_size, f_out, true);
            // utils->printFlatArray3D(ws, batch_size, seq_len, f_out, true);

            // std::cout << "Aafter reshape for multi headed attention" << std::endl;
            // utils->print2DMatrixLastTwo(mhead_out_host, batch_size, n_head, seq_len);
        }
        dLinear = false;
        this->mhead_out_host = mhead_out_host;
        return mhead_out_host;
    }

    // without tranpose it would attend to seq_len which is just tokenized
    // if we swap it with n_head then it would attend to that which has values
    // inillized from he init that as inside of the embeddings.
    void swapHead(bool reverse)
    {
        // Before: Shape(batch_size, seq_len, n_head, d_head)
        // After: Shape(batch_size, n_head, seq_len, d_head)

        // std::cout << "Aafter reshape for multi headed attention" << std::endl;
        // utils->print2DMatrixLastTwo(mhead_out_host, batch_size, n_head, seq_len);

        cudaMemcpy(deviceArrInTranspose, mhead_out_host, batch_size * seq_len * n_head * head_dim * sizeof(float), cudaMemcpyHostToDevice);

        SwapNS(n_head,
               head_dim,
               deviceArrInTranspose,
               mhead_out_device,
               batch_size,
               seq_len,
               reverse);

        cudaMemcpy(mhead_out_host, mhead_out_device, seq_len * batch_size * n_head * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

        // std::cout << "After swapping T" << std::endl;
        // utils->print2DMatrixLastTwo(mhead_out_host, batch_size, n_head, seq_len);

        // very hard to think if in higher dimension, mathematicans cannot imagine higher dimension
        // Now our final resule shape would be Shape(batch_size, n_head, seq_len, d_head)
    }

    float *teansposeKeyForAttnScore(
        float *buffer_out_d
    )
    { // We have a problem here. 

        cudaMemcpy(deviceArrInTranspose, mhead_out_host, batch_size * seq_len * n_head * head_dim * sizeof(float), cudaMemcpyHostToDevice);

        TransposeKey(
            deviceArrInTranspose, // it will replace that same variable variable
            buffer_out_d,
            n_head,
            head_dim,
            batch_size,
            seq_len,
            false);

        cudaMemcpy(mhead_out_host, buffer_out_d, seq_len * batch_size * n_head * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

        return mhead_out_host;
    }
};
