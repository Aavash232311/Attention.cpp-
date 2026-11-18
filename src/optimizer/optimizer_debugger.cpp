//
// Created by avash on 11/18/26.
//
#include <iostream>
#include <cuda_runtime.h>
#include "../include/attention_params.hpp"
#include "optimizer_debugger.hpp"

struct DeviceToHost
{

    float *host_pointer;
    DeviceToHost(float *device_pointer, int N)
    {
        host_pointer = static_cast<float *>(malloc(N * sizeof(float)));
        cudaMemcpy(host_pointer, device_pointer, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    ~DeviceToHost()
    {
        free(host_pointer);
    }
};

/**
 * @class releaseOptimizerHyperparameters
 *
 *
 * @param lr: Learning Rate eta
 * @param wd: Weight decay
 * @param beta_1: beta_1 frictional coefficient
 * @param beta_2: beta_2 frictional coefficient
 * @param epsilon:
 *
 *
 * @brief Releases hyper-parameters for AdamW optimizer
 *
 */
void OptimizerDebugger::releaseOptimizerHyperparameters(int t)
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

// release updated weight at a snap
// release the gradient at a snap
// because that "debug" flag is like a switch
// and it breaks everything

// so what we want is to re-release those weight and
// gradient (freshly calculated gradient pointer)
void OptimizerDebugger::releaseWeightAndGrient(
    FlashAttentionPointers network_parameters,
    AdamWMemConfig &config)
{
    DeviceToHost wq(network_parameters.attention_head.device_WQ, config.d_model * config.d_model);
    DeviceToHost d_wq(network_parameters.d_weight_q, config.d_model * config.d_model);

    bulkRelease<float>(
        {
            {wq.host_pointer, config.d_model * config.d_model, "unoptimal_wq.bin"},
            {d_wq.host_pointer, config.d_model * config.d_model, "inst_grad_q.bin"},
        });
}
