#pragma once

#include "../include/netattention.hpp"
#include "../include/attention_params.hpp"

#include "../include/helper.hpp"
#include "../include/utils.hpp"

#include "./ds/optimizer_mem.hpp"
#include "./ds/AdamW_config.hpp"

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

extern "C" void AdamWSTEP(
    float *theta,
    float *grad,
    float *m,
    float *v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t,
    int n);

using namespace std;

class AdamW
{

public:
    bool debug = false;

    float lr;
    float weight_decay;
    float beta_1;
    float beta_2;
    float epsilon;

    FlashAttentionPointers modelParamaters;

    virtual void releaseOptimizerHyperparameters(int t) {};
    virtual void releaseGrad(
        FlashAttentionPointers modelParamaters,
        AdamWMemConfig &config) {};

    AdamW(
        float lr = 0.01f,
        float weight_decay = 0.0f,
        float beta_1 = 0.09f,
        float beta_2 = 0.09f,
        float epsilon = 1e-8f,
        bool debug = false)
    {
        this->lr = lr;
        this->debug = debug;
        this->beta_1 = beta_1;
        this->beta_2 = beta_2;
        this->epsilon = epsilon;
        this->weight_decay = weight_decay;
    }

    void invoke(
        FlashAttentionPointers modelParamaters,
        AdamWMemConfig &config,
        int epochs)
    {

        if (debug)
        {
            // release hyperparameters
            releaseOptimizerHyperparameters(epochs * config.batch_size);

            /*
                Note:- the sequence does not matter here we go from back to first for all the learnable paramaters, we update them.
            */
        }

        AdamWSTEP(
            modelParamaters.attention_head.device_WQ,
            modelParamaters.d_weight_q, // grad that needs to be updated
            config.dl_dw.m_d,
            config.dl_dw.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model * config.d_model);


        // AdamWSTEP(
        //     modelParamaters.dl_dw_device,
        //     modelParamaters.d_weight_k,
        //     config.dl_dw.m_d,
        //     config.dl_dw.v_d,
        //     lr,
        //     beta_1,
        //     beta_2,
        //     epsilon,
        //     weight_decay,
        //     config.t,
        //     config.d_model * config.d_model);

        releaseGrad(
            modelParamaters,
            config
        );

        config.t++;
    }
};