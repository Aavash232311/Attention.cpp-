#pragma once
#include "linear.hpp"
#include "../include/utils.hpp"
#include "single_embeddings.hpp"

struct AttentionParamaters
{
    LinearParams Q_params;
    LinearParams K_params;
    LinearParams V_params;

    LinearParams Projection;

    LinearParams LayerNorm;

    SingleEmbeddings Embeddings;

    float *S;
    float *P;
    float *O;
    float *V;

    // We are re-using the buffer for BTC and Batch Seq Number of head and head dim to save reources
    float *DEVCIE_BUFFER_BTC;
    float *DEVICE_BUFFER_MULTIHEAD;

    float* Q_cache;
    float* K_cache;

    // shape (B, T, C) after
    // embedding out
    float *x;

    // cache the mean and std-dev shape
    // LayerNorm forward pass cache.
    float *mean_cache;
    float *std_dev_cache;
};