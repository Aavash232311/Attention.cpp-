#pragma once
#include "attention_params.hpp"
#include "linear.hpp"

struct NetAttentionParamaters
{
    AttentionParamaters attention_head;
    LinearParams lm_head;

    /**
     * Cross-entropy loss on the device, one value for each token.
     *
     * @note Expected shape (batch_size, seq_len)
     * @param L location device
     */
    float *L;
    // These pointers let the backward pass use the buffers we already allocated.

    // --------- Variables needed for upstream gradient --------------

    /**
     * MAKE SURE THAT THESE ARE VARIABLES FROM THE DEVICE
     * One-hot targets on the device, shape (B, T, vocab_size).
     *
     * @note Expected shape (batch_size, seq_len, vocab_size)
     * @param y_actual location device
     */
    float *y_actual;
    /**
     * Predicted probabilities, same shape as the targets.
     *
     * @note Expected shape (batch_size, seq_len, vocab_size)
     * @param y_predicted location device
     */
    float *y_predicted;

    /**
     * @note upstream gradient out from softmax + CE
     *
     * @note Expected shape (batch_size, seq_len, vocab_size)
     * @param upsteram_out_softmax_ce location device
     */
    float *upsteram_out_softmax_ce;

    /**
     * LM head weights on the device, shape (C, vocab_size).
     *
     * @note Expected shape (d_model, vocab_size)
     * @param weight_lm_head location device
     */
    float *weight_lm_head;

    /**
     * @note Weight gradient from the LM head, shape (C, vocab_size).
     *
     * @note Expected shape (d_model, vocab_size)
     * @param dweight_lm_head location device
     */
    float *dweight_lm_head;

    /**
     * x = attention(x)
     * @note x is the tensor right after attention layer
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param final_attn_tensor location device
     */
    float *final_attn_tensor;

    /**
     * x = attention(x)
     * xt  = x^t
     * @note x is the tensor right after attention layer
     * and xt is the transposed one on the device.
     *
     * @note Expected shape (batch_size, d_model, seq_len)
     * @param xt_device location device
     */
    float *xt_device;

    /**
     * @brief transposed weight pointer from lm head
     *
     * @note Expected shape (vocab_size, d_model)
     * @param wt_lm_head_deice location device
     */
    float *wt_lm_head_deice;
};

struct FlashAttentionPointers : NetAttentionParamaters
{

    /**
     * We keep P_T and V_T in their own buffers so the backward pass can use them.
     *
     * @note Expected shape (batch_size, num_heads, seq_len, seq_len)
     * @param P_T_device_out location device
     */
    float *P_T_device_out;

    /**
     * @brief Value matrix transposed to
     * Org  : (B, n_head, T, head_dim)
     * Shape: (B, n_head, head_dim, T)
     * @note shape (B, n_head, head_dim, T)
     *
     * @note Expected shape (batch_size, num_heads, head_dim, seq_len)
     * @param V_T_device_out location device
     */
    float *V_T_device_out;

    /**
     * @brief Upstream gradient from output projection
     * reshaped to (B, num_head, seq_len, head_dim)
     * @note shape (B, num_head, seq_len, head_dim)
     *
     * @note Expected shape (batch_size, num_heads, seq_len, head_dim)
     * @param Uncontact_G_Upstream location device
     */
    float *Uncontact_G_Upstream;

    /**
     * @brief Upstream gradient from output projection
     * @note shape (B, T, C)
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param upstream_grad_output_proj location device
     */
    float *upstream_grad_output_proj;

    /**
     * @brief Upstream gradient from softmax.
     * @note (B, H, T, T)
     *
     * @note Expected shape (batch_size, num_heads, seq_len, seq_len)
     * @param d_softmax_upstream location device
     */
    float *d_softmax_upstream;

    /**
     * Gradient for V, shape (B, H, T, D).
     *
     * @note Expected shape (batch_size, num_heads, seq_len, head_dim)
     * @param dV location device
     */
    float *dV;
    /**
     * Gradient for the attention probabilities, shape (B, H, T, T).
     *
     * @note Expected shape (batch_size, num_heads, seq_len, seq_len)
     * @param dP location device
     */
    float *dP;

    /**
     * Extra device scratch buffer passed in by the model.
     *
     * @note Expected shape (batch_size, num_heads, seq_len, seq_len)
     * @param ppt location device
     */
    float *ppt;

    /**
     * Gradient for Q, shape (B, H, T, D).
     *
     * @note Expected shape (batch_size, num_heads, seq_len, head_dim)
     * @param dQ location device
     */
    float *dQ;
    /**
     * Gradient for K, same shape as dQ.
     *
     * @note Expected shape (batch_size, num_heads, seq_len, head_dim)
     * @param dK location device
     */
    float *dK;

    /**
     * LayerNorm input gradient before adding the residual gradient.
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param dx location device
     */
    float *dx;

    /**
     * @brief transpose of upstream gradient from d softmax
     * @note (B, H, T, T)
     *
     * @note Expected shape (batch_size, num_heads, seq_len, seq_len)
     * @param d_score_t location device
     */
    float *d_score_t;

    // weights of Q, K and V
    // These are in device from linear

    /**
     * Transposed K weights from the linear layer.
     *
     * @note Expected shape (d_model, d_model)
     * @param WkT location device
     */
    float *WkT;
    /**
     * Transposed V weights from the linear layer.
     *
     * @note Expected shape (d_model, d_model)
     * @param WvT location device
     */
    float *WvT;
    /**
     * Transposed Q weights. The LM head has a different shape; I got confused here before.
     *
     * @note Expected shape (d_model, d_model)
     * @param WqT location device
     */
    float *WqT;

    /**
     * K weights from the linear layer.
     *
     * @note Expected shape (d_model, d_model)
     * @param Wk location device
     */
    float *Wk;
    /**
     * V weights from the linear layer.
     *
     * @note Expected shape (d_model, d_model)
     * @param WV location device
     */
    float *WV;
    /**
     * Q weights from the linear layer.
     *
     * @note Expected shape (d_model, d_model)
     * @param WQ location device
     */
    float *WQ;

    // Correctness first optimization later, already very very complicated

    // G_x_hat

    /**
     * Merge the dQ heads from (B, H, T, D) back into (B, T, C).
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param qUp location device
     */
    float *qUp;

    /**
     * Merge the dK heads from (B, H, T, D) back into (B, T, C).
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param kUp location device
     */
    float *kUp;

    /**
     * Merge the dV heads from (B, H, T, D) back into (B, T, C).
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param vUp location device
     */
    float *vUp;

    // resultant value of
    // dV wT, dQ wT, dK wT you get it.

    /**
     * dQ times the transposed Q weights, back to the input channels.
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param dqWt location device
     */
    float *dqWt;
    /**
     * Same idea for K, dK times the transposed K weights.
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param dkWt location device
     */
    float *dkWt;
    /**
     * Same idea for V, dV times the transposed V weights.
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param dvWt location device
     */
    float *dvWt;

    /**
     * G_x_hat
     * Add the Q, K and V input gradients here before LayerNorm backward.
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param G_x_hat location device
     */
    float *G_x_hat;

    /**
     * LayerNorm bias gradient, one value per channel.
     *
     * @note Expected shape (d_model)
     * @param debeta location device
     */
    float *debeta;
    /**
     * LayerNorm scale gradient, one value per channel.
     *
     * @note Expected shape (d_model)
     * @param dgamma location device
     */
    float *dgamma;

    /**
     * @brief gradient of bias from lm head (dbias lm head)
     *
     * @note Expected shape (vocab_size)
     * @param dbias_lm_head_pred location device
     */
    float *dbias_lm_head_pred;

    /**
     * LM head bias, one value for each character in the vocab.
     *
     * @note Expected shape (vocab_size)
     * @param bias_lm_head location device
     */
    float *bias_lm_head;

    /**
     * Transposed output projection weights, shape (C, C).
     *
     * @note Expected shape (d_model, d_model)
     * @param WoT location device
     */
    float *WoT;

    /**
     * @brief Upstream gradient from the LM head linear layer.
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param dl_dh_output location device
     */
    float *dl_dh_output;

    /**
     * Add the residual path back to dx here, shape (B, T, C).
     *
     * @note Expected shape (batch_size, seq_len, d_model)
     * @param d_add_residual_output location device
     */
    float *d_add_residual_output;

    /**
     * Token embedding table gradient, shape (vocab_size, C).
     *
     * @note Expected shape (vocab_size, d_model)
     * @param d_embedding location device
     */
    float *d_embedding;

    /**
     * Bias gradient for Q, one value per channel.
     *
     * @note Expected shape (d_model)
     * @param d_bias_q location device
     */
    float *d_bias_q;
    /**
     * Bias gradient for K, one value per channel.
     *
     * @note Expected shape (d_model)
     * @param d_bias_k location device
     */
    float *d_bias_k;
    /**
     * Bias gradient for V, one value per channel.
     *
     * @note Expected shape (d_model)
     * @param d_bias_v location device
     */
    float *d_bias_v;

    /**
     * Weight gradient for Q, shape (C, C).
     *
     * @note Expected shape (d_model, d_model)
     * @param d_weight_q location device
     */
    float *d_weight_q;
    /**
     * Weight gradient for K, shape (C, C).
     *
     * @note Expected shape (d_model, d_model)
     * @param d_weight_k location device
     */
    float *d_weight_k;
    /**
     * Weight gradient for V, shape (C, C).
     *
     * @note Expected shape (d_model, d_model)
     * @param d_weight_v location device
     */
    float *d_weight_v;

    /**
     * Output projection weight gradient, shape (C, C).
     *
     * @note Expected shape (d_model, d_model)
     * @param d_weight_output_project location device
     */
    float *d_weight_output_project;

    /**
     * Token IDs for this batch of text.
     * It's basically a chunk of the text, with each character encoded as an integer.
     *
     * @note Expected shape (batch_size, seq_len)
     * @param token_id location device
     */
    int *token_id;
};
