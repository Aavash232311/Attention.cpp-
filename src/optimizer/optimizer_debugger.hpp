//
// Created by Avash on 11/18/26.
//
#ifndef CHARACTER_TRANSFORMER_OPTIMIZER_DEBUGGER_H
#define CHARACTER_TRANSFORMER_OPTIMIZER_DEBUGGER_H


#include <cuda_runtime.h>
#include <iostream>

#include "./AdamW.hpp"


#include "../include/cache_out.hpp"
#include "../include/netattention.hpp"
#include "../include/attention_params.hpp"

using namespace std;


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

    friend  void releaseOptimizerHyperparameters();
    void releaseOptimizerHyperparameters(int t) override;
    void releaseWeightAndGrient(FlashAttentionPointers network_parameters, AdamWMemConfig &config) override;
};

#endif //CHARACTER_TRANSFORMER_OPTIMIZER_DEBUGGER_H
