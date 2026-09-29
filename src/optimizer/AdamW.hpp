#pragma once

#include "../include/netattention.hpp"
#include "../include/attention_params.hpp"

#include "../include/helper.hpp"
#include "../include/utils.hpp"

#include "./ds/optimizer_mem.hpp"

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
    // AdamWMemConfig gradient_optimizer_properties;

    virtual void releaseOptimizerHyperparameters() {};

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

    void invoke(FlashAttentionPointers modelParamaters)
    {
        if (debug)
        {
            // release hyperparameters
            releaseOptimizerHyperparameters();

            /*
                Note:- the sequence does not matter here we go from back to first for all the learnable paramaters, we update them.
            */
        }

        // AdamWSTEP(
        //     modelParamaters.dl_dz_out_device
        //     modelParamaters.dl_dz_out_device,

        // );
    }
};