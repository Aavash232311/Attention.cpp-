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


using namespace std;

class OptimizerDebugger: public AdamW {
    public:
        OptimizerDebugger(
            float lr,
            float weight_decay,
            float beta_1,
            float beta_2,
            float epsilon,
            bool debug
        )
        : AdamW(
            lr,
            weight_decay,
            beta_1,
            beta_2,
            epsilon,
            debug
        )
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
    void releaseOptimizerHyperparameters()
    {
        vector<string> key = {
            "lr",
            "wd",
            "beta_1",
            "beta_2",
            "epsilon"  
        };

        vector<float> value = {
            lr,
            weight_decay,
            beta_1,
            beta_2,
        };

        releaseConfig(
            key,
            value,
            "./src/cache/optimizer_config.json"
        );
    }
};