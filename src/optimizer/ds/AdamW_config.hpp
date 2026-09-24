#pragma once
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./optimizer_mem.hpp"

struct AdamWMemConfig
{
public:
    // For interface back. Ex: these are the local gradients
    AdamWPhysics dl_dw;
    AdamWPhysics dl_db;

    // For attention head 
    AdamWPhysics weight_output_proj;
    AdamWPhysics bias_output_proj;

    AdamWPhysics dQ;
    AdamWPhysics dK;
    AdamWPhysics dV;

    AdamWPhysics Ln_gamma;
    AdamWPhysics Ln_beta;

    AdamWPhysics d_token_embedding;

    AdamWPhysics d_weight_q;
    AdamWPhysics d_weight_k;
    AdamWPhysics d_weight_v;

    AdamWPhysics d_bias_q;
    AdamWPhysics d_bias_k;
    AdamWPhysics d_bias_v;

    int batch_size;
    int seq_len;
    int d_model;
    int vocab_size;
    int num_heads;

    AdamWMemConfig(
        int batch_size,
        int seq_len,
        int d_model,
        int vocab_size,
        int num_heads
    )
    {
        this->batch_size = batch_size;
        this->d_model = d_model;
        this->seq_len = seq_len;
        this->vocab_size = vocab_size;
        this->num_heads = num_heads;

        dl_dw.init(vocab_size * d_model);
        dl_db.init(vocab_size);

        weight_output_proj.init(d_model * d_model);
        bias_output_proj.init(d_model);

        dQ.init(d_model * d_model);
        dK.init(d_model * d_model);
        dV.init(d_model * d_model);

        Ln_gamma.init(d_model);
        Ln_beta.init(d_model);

        d_token_embedding.init(vocab_size * d_model);

        d_weight_q.init(d_model * d_model);
        d_weight_k.init(d_model * d_model);
        d_weight_v.init(d_model * d_model);

        d_bias_q.init(d_model);
        d_bias_k.init(d_model);
        d_bias_v.init(d_model);
    }
};