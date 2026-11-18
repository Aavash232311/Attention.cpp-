#pragma once

#include "../include/netattention.hpp"

#include "../include/utils.hpp"
#include "./ds/AdamW_config.hpp"
#include <memory>

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
    virtual ~AdamW() = default;

    bool debug = true;

    float lr;
    float weight_decay;
    float beta_1;
    float beta_2;
    float epsilon;

    FlashAttentionPointers network_parms{};

    virtual void releaseOptimizerHyperparameters(int t) {};
    virtual void releaseOptimizedWeight(
        FlashAttentionPointers modelParamaters,
        AdamWMemConfig &config) {};

    virtual void releaseWeightAndGrient(
        FlashAttentionPointers modelParamaters,
        AdamWMemConfig &config) {};

    std::unique_ptr<Utility> utils = std::make_unique<Utility>();

    AdamW(
        float lr = 0.01f,
        float weight_decay = 0.0f,
        float beta_1 = 0.9f,
        float beta_2 = 0.999f,
        float epsilon = 1e-8f,
        bool debug = false);

    void invoke(
        FlashAttentionPointers modelParamaters,
        AdamWMemConfig &config,
        int epochs);
};