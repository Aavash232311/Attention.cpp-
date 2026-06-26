#include "linear.hpp"
#include "single_embeddings.hpp"

#pragma once
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

    // We are re-using the buffer for BTC and Batch Seq Number of head and head dim to save reources
    float *DEVCIE_BUFFER_BTC;
    float *DEVICE_BUFFER_MULTIHEAD;
};