#pragma once
#include "linear.hpp"
#include "../include/utils.hpp"
#include "single_embeddings.hpp"

// I am afraid that I manage memory like I have no common sense at all
// but anyway right now we just want it to work.

// todo: add a production grade comment after a single working prototype comes into play
// todo: add a production grade comment after a single working prototype comes into play
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

    float *Q_cache;
    float *K_cache;

    /**
     * (B,T,C) tensor after net embedding added location GPU
     *
     * @param x (BTC) tensor inside of the GPU, after net embedding.
     */
    float *x;

    /**
     * mean cache from forward pass LayerNorm
     * Shape (B * T, C)
     * @param mean_cache location device
     */
    float *mean_cache;

    /**
     * std dev cache from forward pass LayerNorm
     * Shape (B * T, C)
     * @param std_dev_cache location device
     */
    float *std_dev_cache;

    /**
     * local derivative of learnable paramater gamma
     * Shape (C)
     * @param d_gamma device
     */
    float *d_gamma;

    /**
     * local derivative of learnable paramater beta
     * Shape (C)
     * @param d_beta device
     */
    float *d_beta;

    /**
     * Weight of query
     * Shape (C, C)
     * @param device_WQ device
     */
    float *device_WQ;
    /**
     * Weight of key
     * Shape (C, C)
     * @param device_WK device
     */
    float *device_WK;
    /**
     * Weight of value
     * Shape (C, C)
     * @param device_WV device
     */
    float *device_WV;

    /**
     * concatenation paramater (output projection) weight
     * Shape (B, T, C)
     * @param wo location device
     */
    float *wo;

    /**
     * concatenation paramater (output projection) bias
     * Shape (C, )
     *
     * @param wo_bias location device
     */
    float *wo_bias;

    /**
     * concatenation paramater (output projection) bias output
     * dbias, sum across B, T dimension of the upstream gradient of shape (B, T, C)
     * Shape (C)
     *
     * @param wo_bias location device
     */
    float *doutput_bias;
};