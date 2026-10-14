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
extern "C" void multiHeadedAttention(float *ws, float *out, int num_head, int head_dimension, int batch_size, int d_model, int seq_len);
extern "C" void TransposeKey(float *arr, float *out, int num_heads, int head_dim, int batch_size, int seq_len, bool reverse);
extern "C" void TranposeMiddleTensor4(float *arr, float *out, int batch_size, int seq_len, int num_head, int head_dim);

class Linear
{
    bool dLinear = true;
    float *ws = nullptr; // weighted sum

    float *x = nullptr;

    float *h_mha;

    float *d_x;
    float *d_w;
    float *d_b;
    float *weighted_sum_device;

    int f_in;
    int f_out;
    int seq_len;
    int batch_size;
    std::unique_ptr<Utility> utils = std::make_unique<Utility>();

    float *mhead_out_device; // this is multi head attention out device
    float *device_hhead_in;  // copy to this from weighted sum

    float n_head;
    float head_dim;

    bool debug = true;

    float *deviceArrInTranspose;

    float *device_weight;
    float *device_bias;
    curandState *d_state_weight;
    curandState *d_state_bias;

public:
    float *weight = nullptr;
    float *bias = nullptr;

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
        cudaMalloc((void **)&weighted_sum_device, seq_len * batch_size * f_out * sizeof(float));

        // copy to the device, those allocated weight and biases.
        cudaMemcpy(d_w, weight, f_in * f_out * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, bias, f_out * sizeof(float), cudaMemcpyHostToDevice);

        /// allocate the memory for multi headed attention Shape(B, T, num_heads, head_dim), JUST BTC when in
        cudaMalloc((void **)&device_hhead_in, seq_len * batch_size * f_out * sizeof(float)); // copy ws here

        cudaMalloc((void **)&mhead_out_device, seq_len * batch_size * n_head * head_dim * sizeof(float));

        // In order to transpose that Q,K,V and swap we need to allocate the memory in GDDR VRAM
        cudaMalloc((void **)&deviceArrInTranspose, batch_size * seq_len * n_head * head_dim * sizeof(float));
    }

    ~Linear()
    {
        free(ws);
        free(weight);
        free(bias);

        cudaFree(d_x);
        cudaFree(d_w);
        cudaFree(d_b);
        cudaFree(weighted_sum_device);

        cudaFree(deviceArrInTranspose);

        cudaFree(mhead_out_device);

        cudaFree(device_weight);
        cudaFree(device_bias);
        cudaFree(d_state_weight);
        cudaFree(d_state_bias);
    }

    // this is the exntension of the constructor dont know why this way but
    void LinearParams(int fan_in, int fan_out)
    {

        weight = (float *)malloc(fan_in * fan_out * sizeof(float)); // these are out for both weight and biases.
        bias = (float *)malloc(fan_out * sizeof(float));

        cudaMalloc((void **)&device_weight, fan_in * fan_out * sizeof(float));
        cudaMalloc(&d_state_weight, fan_in * fan_out * sizeof(curandState));
        cudaMalloc(&d_state_bias, fan_out * sizeof(curandState));
        cudaMalloc((void **)&device_bias, fan_out * sizeof(float));

        // making sure these seeds are different
        unsigned long seed_1 = static_cast<unsigned long>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count());

        unsigned long seed_2 = static_cast<unsigned long>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count());

        // we need tow kernal launches here
        KaimingInit(device_weight, d_state_weight, fan_in, fan_out, seed_1);
        KaimingInit(device_bias, d_state_bias, 1, fan_out, seed_2);

        cudaMemcpy(weight, device_weight, fan_in * fan_out * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bias, device_bias, fan_out * sizeof(float), cudaMemcpyDeviceToHost);

        // if (!(debug))
        // {
        //     std::cout << "Weight" << std::endl;
        //     utils->printFlatArray2D(weight, fan_in, fan_out);
        //     std::cout << "Bias" << std::endl;
        //     utils->printFlatArray2D(this->bias, fan_out, 1);
        //     std::cout << "Fan in: " << fan_in << " Fan out: " << fan_out << std::endl;
        // }
    }

    float *getBias()
    {
        // get bias from derive
        return this->bias;
    }

    float *getBaiasDevice()
    {
        return this->d_b;
    }

    float *getWeight()
    {
        return this->weight;
    }

    float *getWeightDevice()
    {
        return this->device_weight;
    }

    float *forward(float *val)
    {
        this->x = val;
        int m = batch_size * seq_len;

        cudaMemcpy(d_x, x, seq_len * batch_size * f_in * sizeof(float), cudaMemcpyHostToDevice);

        WeightedSum(d_x, d_w, d_b, weighted_sum_device, m, f_in, f_out);

        cudaMemcpy(ws, weighted_sum_device, seq_len * batch_size * f_out * sizeof(float), cudaMemcpyDeviceToHost);

        debug =

            dLinear = false;

        return weighted_sum_device;
    }

    float *forward(float *val, std::string type)
    {
        WeightedSum(val, d_w, d_b, weighted_sum_device, batch_size * seq_len, f_in, f_out);
        return weighted_sum_device;
    }

    float *getMultiHeadedDeivce()
    {
        return this->mhead_out_device;
    }

    /**
     * @brief turns (B,T,C) tensor into (batch_size, seq_len, num_head, head_dim)
     *
     * @warning Returns device pointer
     * @return (batch_size, seq_len, num_head, head_dim) tensor in device
     */
    float *reshapeHead()
    {
        multiHeadedAttention(weighted_sum_device, mhead_out_device, n_head, head_dim, batch_size, f_out, seq_len);

        return mhead_out_device;
    }

    float *transposeMiddle4DTensor()
    {
        TranposeMiddleTensor4(
            mhead_out_device,
            deviceArrInTranspose,
            batch_size,
            seq_len,
            n_head,
            head_dim);

        return deviceArrInTranspose; // (batch_size, num_head, seq_len, head_dim) changed shape
    }

    /**
     * @class teansposeKeyForAttnScore
     * @brief transforms (batch_size, num_head, seq_len, head_dim) to (batch_size, num_head, head_dim, seq_len)
     *
     * @warning Returns device pointer
     * @return (batch_size, seq_len, num_head, head_dim) tensor in device
     */
    void teansposeKeyForAttnScore(
        float *buffer_out_d)
    {
        TransposeKey(
            deviceArrInTranspose,
            buffer_out_d,
            n_head,
            head_dim,
            batch_size,
            seq_len,
            false);
    }
};
