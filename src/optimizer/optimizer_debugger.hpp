#pragma once
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./AdamW.hpp"

#include "../include/utils.hpp"
#include "../include/linear.hpp"
#include "../include/p_head.hpp"
#include "../include/cache_in.hpp"
#include "../include/cache_out.hpp"
#include "../include/netattention.hpp"
#include "../include/attention_params.hpp"
#include "../include/single_embeddings.hpp"

#include "../include/cache_out.hpp"

using namespace std;

struct DeviceToHost
{

    float *host_pointer;
    DeviceToHost(float *device_pointer, int N)
    {
        host_pointer = (float *)malloc(N * sizeof(float));
        cudaMemcpy(host_pointer, device_pointer, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    ~DeviceToHost()
    {
        free(host_pointer);
    }
};

class OptimizerDebugger : public AdamW
{
public:
    OptimizerDebugger(
        float lr,
        float weight_decay,
        float beta_1,
        float beta_2,
        float epsilon,
        bool debug)
        : AdamW(
              lr,
              weight_decay,
              beta_1,
              beta_2,
              epsilon,
              debug)
    {
    }

    /**
     * @class releaseOptimizerHyperparameters
     *
     *
     * @param lr: Learning Rate eta
     * @param wd: Weight decay
     * @param beta_1: beta_1 frictional cofficient
     * @param beta_2: beta_2 frictional cofficient
     * @param epsilon:
     *
     *
     * @brief Releases hyper-paramaters for AdamW optimizer
     *
     */
    void releaseOptimizerHyperparameters(int t)
    {
        vector<string> key = {
            "lr",
            "wd",
            "beta_1",
            "beta_2",
            "epsilon",
            "t"};

        vector<float> value = {
            lr,
            weight_decay,
            beta_1,
            beta_2,
            epsilon,
            static_cast<float>(t)};

        releaseConfig(
            key,
            value,
            "./src/cache/optimizer_config.json");
    }

    void releaseGrad(
        FlashAttentionPointers modelParamaters,
        AdamWMemConfig &config)
    {
        DeviceToHost d_weight_q(modelParamaters.attention_head.device_WQ, config.d_model * config.d_model);

        bulkRelease<float>(
            {
                {d_weight_q.host_pointer, config.d_model * config.d_model, "d_weight_q_optimal.bin"},
            });
    }
};