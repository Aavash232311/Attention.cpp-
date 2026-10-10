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

        // lm head weight and bias

        // weight for lm head
        AdamWSTEP(
            modelParamaters.weight_lm_head,
            modelParamaters.dweight_lm_head, // grad that needs to be updated
            config.dl_dw.m_d,
            config.dl_dw.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model * config.vocab_size);

        // bias for lm head
        AdamWSTEP(
            modelParamaters.bias_lm_head,
            modelParamaters.dbias_lm_head_pred,
            config.dl_db.m_d,
            config.dl_db.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.vocab_size);

        // output projection linear layer

        // weight
        AdamWSTEP(
            modelParamaters.attention_head.wo,
            modelParamaters.d_weight_output_project,
            config.weight_output_proj.m_d,
            config.weight_output_proj.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model * config.d_model);

        // bias
        AdamWSTEP(
            modelParamaters.attention_head.wo_bias,
            modelParamaters.attention_head.doutput_bias,
            config.bias_output_proj.m_d,
            config.bias_output_proj.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model);

        // QKV update

        // Q weight

        AdamWSTEP(
            modelParamaters.attention_head.device_WQ,
            modelParamaters.d_weight_q,
            config.d_weight_q.m_d,
            config.d_weight_q.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model * config.d_model);

        // Q bias

        AdamWSTEP(
            modelParamaters.attention_head.bias_q,
            modelParamaters.d_bias_q,
            config.d_bias_q.m_d,
            config.d_bias_q.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model);

        // K wieght
        AdamWSTEP(
            modelParamaters.attention_head.device_WK,
            modelParamaters.d_weight_k,
            config.d_weight_k.m_d,
            config.d_weight_k.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model * config.d_model);

        // K bias
        AdamWSTEP(
            modelParamaters.attention_head.bias_k,
            modelParamaters.d_bias_k,
            config.d_bias_k.m_d,
            config.d_bias_k.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model);

        // v weight
        AdamWSTEP(
            modelParamaters.attention_head.device_WV,
            modelParamaters.d_weight_v,
            config.d_weight_v.m_d,
            config.d_weight_v.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model * config.d_model);

        // v bias
        AdamWSTEP(
            modelParamaters.attention_head.bias_v,
            modelParamaters.d_bias_v,
            config.d_bias_v.m_d,
            config.d_bias_v.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model);

        // gamma and beta of the normalization paramater

        // learning paramater gamma
        AdamWSTEP(
            modelParamaters.attention_head.gamma, // this is the gamma pointer I will change this conflicting name
            modelParamaters.dgamma,
            config.Ln_gamma.m_d,
            config.Ln_gamma.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model);

        // learning paramater beta
        AdamWSTEP(
            modelParamaters.attention_head.beta, // this is the gamma pointer I will change this conflicting name
            modelParamaters.debeta,
            config.Ln_beta.m_d,
            config.Ln_beta.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.d_model);

        // now "one" of the embedding only "one" we are using the traditional transformer

        AdamWSTEP(
            modelParamaters.attention_head.device_embedding_paramaters, // this is the gamma pointer I will change this conflicting name
            modelParamaters.d_embedding,
            config.d_token_embedding.m_d,
            config.d_token_embedding.v_d,
            lr,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            config.t,
            config.vocab_size * config.d_model);

        cudaDeviceSynchronize();

        config.t++;
    }
};