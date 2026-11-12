#pragma once
#include "linear.hpp"
#include "../include/utils.hpp"
#include "single_embeddings.hpp"

// I am afraid that I manage memory like I have no common sense at all
// but anyway right now we just want it to work.

struct AttentionParamaters
{


    /**
     * Attention score pointer. Kept here for the attention cache.
     *
     * @note No active shape yet; this pointer is currently unused.
     * @param S location not assigned in the current implementation
     */
    float *S;

    /**
     * Softmaxed value in the process of attention score.
     * @note Shape (batch_size, num_heads, seq_len, seq_len)
     *
     * @note Expected shape (batch_size, num_heads, seq_len, seq_len)
     * @param P location device
     */
    float *P;
    /**
     * Attention output pointer. Kept here for the attention cache.
     *
     * @note No active shape yet; this pointer is currently unused.
     * @param O location not assigned in the current implementation
     */
    float *O;

    /**
     * Value matrix
     * @note Shape (batch_size, num_heads, seq_len, head_dim)
     *
     * @note Expected shape (batch_size, num_heads, seq_len, head_dim)
     * @param V location device
     */
    float *V;

    /**
     * We are re-using the buffer for BTC and Batch Seq Number of head and head dim to save resources
     * Scratch space for (B, T, C) tensors on the device.
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param DEVCIE_BUFFER_BTC location device
     */
    float *DEVCIE_BUFFER_BTC;
    /**
     * Scratch space for (B, H, T, D) tensors on the device.
     *
     * @note Expected shape (batch_size, num_heads, seq_len, head_dim)
     * @param DEVICE_BUFFER_MULTIHEAD location device
     */
    float *DEVICE_BUFFER_MULTIHEAD;

    /**
     * Keep Q for backward so we do not have to calculate it again, shape (B, H, T, D).
     *
     * @note Expected shape (batch_size, num_heads, seq_len, head_dim)
     * @param Q_cache location device
     */
    float *Q_cache;
    /**
     * Same idea for K, shape (B, H, T, D).
     *
     * @note Expected shape (batch_size, num_heads, seq_len, head_dim)
     * @param K_cache location device
     */
    float *K_cache;

    /**
     * (B,T,C) tensor after net embedding added location GPU
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param x location device
     */
    float *x;

    /**
     * Normalized input transposed from (B, T, C) to (B, C, T)
     *
     * @note Expected shape (batch_size, d_model, seq_len)
     * @param normalized_xt location device
     */
    float *normalized_xt;

    /**
     * (B,T,C) tensor after the layer norm on the forward pass
     * @note this goes in as "x" in the QKV linear parameters.
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param normalized_x location device
     */
    float *normalized_x;

    /**
     * mean cache from forward pass LayerNorm
     * Shape (B * T), one value per token
     *
     * @note Expected shape (batch_size * seq_len)
     * @param mean_cache location device
     */
    float *mean_cache;

    /**
     * std dev cache from forward pass LayerNorm
     * Shape (B * T), one value per token
     *
     * @note Expected shape (batch_size * seq_len)
     * @param std_dev_cache location device
     */
    float *std_dev_cache;

    /**
     * Learnable LayerNorm scale gamma, not its gradient
     * Shape (C)
     *
     * @note Expected shape (d_model)
     * @param gamma location device
     */
    float *gamma;

    /**
     * Learnable LayerNorm bias beta, not its gradient
     * Shape (C)
     *
     * @note Expected shape (d_model)
     * @param beta location device
     */
    float *beta;

    /**
     * Weight of query
     * Shape (C, C)
     *
     * @note Expected shape (d_model, d_model)
     * @param device_WQ location device
     */
    float *device_WQ;
    /**
     * Weight of key
     * Shape (C, C)
     *
     * @note Expected shape (d_model, d_model)
     * @param device_WK location device
     */
    float *device_WK;
    /**
     * Weight of value
     * Shape (C, C)
     *
     * @note Expected shape (d_model, d_model)
     * @param device_WV location device
     */
    float *device_WV;

    /**
     * Output projection weight after merging the heads
     * Shape (C, C)
     *
     * @note Expected shape (d_model, d_model)
     * @param wo location device
     */
    float *wo;

    /**
     * concatenation parameter (output projection) bias
     * Shape (C, )
     *
     * @note Expected shape (d_model)
     * @param wo_bias location device
     */
    float *wo_bias;

    /**
     * concatenation parameter (output projection) bias output
     * dbias, sum across B, T dimension of the upstream gradient of shape (B, T, C)
     * Shape (C)
     *
     * @note Expected shape (d_model)
     * @param doutput_bias location device
     */
    float *doutput_bias;

    /**
     * Bias of the query from attention head.
     *
     * @note Expected shape (d_model)
     * @param bias_q location device
     */
    float *bias_q;
    /**
     * Bias of the key from attention head.
     *
     * @note Expected shape (d_model)
     * @param bias_k location device
     */
    float *bias_k;
    /**
     * Bias of the value from attention head.
     *
     * @note Expected shape (d_model)
     * @param bias_v location device
     */
    float *bias_v;

    /**
     * Output projection cache x for backpropagation
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param output_porjection_x location device
     */
    float *output_porjection_x;

    /**
     * For local gradient weight wt we need this for output project
     *
     * @note Expected shape (batch_size, d_model, seq_len)
     * @param output_project_xt location device
     */
    float *output_project_xt;

    /**
     * The actual token embedding table, originally generated on the host and now on the device.
     * This is the "theta" that AdamW updates.
     *
     * @note Expected shape (vocab_size, d_model)
     * @param device_embedding_paramaters location device
     */
    float *device_embedding_paramaters;
};
