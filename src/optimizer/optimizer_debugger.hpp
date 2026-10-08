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

    void releaseOptimizedWeight(
        FlashAttentionPointers modelParamaters,
        AdamWMemConfig &config)
    {
        DeviceToHost d_weight_q(modelParamaters.attention_head.device_WQ, config.d_model * config.d_model);

        DeviceToHost m_d(config.dQ.m_d, config.d_model * config.d_model);
        DeviceToHost v_d(config.dQ.v_d, config.d_model * config.d_model);

        bulkRelease<float>(
            {{d_weight_q.host_pointer, config.d_model * config.d_model, "d_weight_q_optimal.bin"},
             {m_d.host_pointer, config.d_model * config.d_model, "m_d.bin"},
             {v_d.host_pointer, config.d_model * config.d_model, "v_d.bin"}});
    }

    // release updated weight at a snap
    // release the gradient at a snap
    // because that "debug" flag is like a switch
    // and it breaks everything

    // so what we want is to re-release those weight and
    // gradient (freshly calculated gradeint pointer)

    void releaseWeightAndGrient(
        FlashAttentionPointers modelParamaters,
        AdamWMemConfig &config)
    {
        DeviceToHost wq(modelParamaters.attention_head.device_WQ, config.d_model * config.d_model);
        DeviceToHost d_wq(modelParamaters.d_weight_q, config.d_model * config.d_model);

        bulkRelease<float>(
            {
                {wq.host_pointer, config.d_model * config.d_model, "unoptimal_wq.bin"},
                {d_wq.host_pointer, config.d_model * config.d_model, "inst_grad_q.bin"},
            });
    }
};