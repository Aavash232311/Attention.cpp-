#pragma once
#include "attention_params.hpp"
#include "linear.hpp"

struct NetAttentionParamaters
{
    AttentionParamaters attention_head;
    LinearParams lm_head;

    float *L; // loss from the cross entropy loss starting of the backpropagation
    // so the above struct contains a pointer reference for CPU memory but we need to make a buffer for gpu memory right here.

    // --------- Variables needed for upstream gradient --------------

    // MAKE SURE THAT THSE ARE VARIABLES FROM THE DEVICE
    float *y_actual;
    float *y_predicted;
    float *dl_dz_out_device;
    float *dl_dz_out_host; // upstream gradient dl/dz

    float *weight_lm_head;

    /**
     *
     * @note weight gradient form lm head.
     *
     * @param x location host
     * @warning location is at the HOST
     */
    float *dweight_lm_head;

    /**
     * x = attention(x)
     *
     * @note x is the tensor right after attenton layer
     *
     * @param x location host
     * @warning location is at the HOST
     */
    float *x_host;

    /**
     * x = attention(x)
     *
     * @note x is the tensor right after attenton layer
     *
     * @param x location device
     * @warning location is at the device
     */
    float *x;

    /**
     * x = attention(x)
     * xt  = x^t
     *
     * @note x is the tensor right after attenton layer
     *      and xt is the transposed once in device.
     * @param xt_device location device
     */
    float *xt_device;

    /**
     * @brief transposed weight pointer from lm head
     *
     * @param wt_lm_head_deice location device
     */
    float *wt_lm_head_deice;
};

struct FlashAttentionPointers : NetAttentionParamaters
{

    /**
     * @brief Transposed P = softmax() from the attention procress
     *
     * @note shape (B, num_head, seq_len, head_dim)
     *
     * @concept device to device copy in attention head
     * 
     * @param P_T_device location device
     */
    float *P_T_device;

    // wewe P_T and V_T to be output such that we can consume the passed arr
    float *P_T_device_out;


    /**
     * @brief Value matrix transposed to
     * 
     * Org  : (B, n_head, T, head_dim)
     * Shape: (B, n_head, head_dim, T)
     *
     * @note shape (B, n_head, head_dim, T)
     *
     * @param V_T_device_out location device
     */
    float *V_T_device_out;

    /**
     * @brief Upstream gradient from output projection
     * re-shapped to (B, num_head, seq_len, head_dim)
     *
     * @note shape (B, num_head, seq_len, head_dim)
     *
     * @param Uncontact_G_Upstream location device
     */
    float *Uncontact_G_Upstream;

    /**
     * @brief Upstream gradient from output projection
     *
     * @note shape (B, T, C)
     *
     * @param Contact_G_Upstream location device
     */
    float *upstream_grad_output_proj;

    float *dV;
    float *dP;

    float *ppt;

    float *dQ;
    float *dK;

    float *d_score_t;

    // weights of Q, K and V
    // These are in device from linear

    float *WkT;
    float *WvT;
    float *WqT; // shape in LM head and shape of in attention head is not the same I got confused

    float *Wk;
    float *WV;
    float *WQ;

    // Correctness first optimization layer, already very very complicated

    // G_x_hat

    /**
     * Reshaped weight dQ from the attention head from (batch_size, seq_len, num_head, head_dim) to (b,t,c)
     * @param qUp location device
     */
    float *qUp;

    /**
     * Reshaped weight dQ from the attention head from (batch_size, seq_len, num_head, head_dim) to (b,t,c)
     * @param kUp location device
     */
    float *kUp;

    /**
     * Reshaped weight dQ from the attention head from (batch_size, seq_len, num_head, head_dim) to (b,t,c)
     * @param vUp location device
     */
    float *vUp;

    // resultant value of
    // dV wT, dQ wT, dK wT you get it.

    float *dqWt;
    float *dkWt;
    float *dvWt;

    // G_x_hat
    float *G_x_hat;

    float *debeta;
    float *dgamma;

    /**
     * @brief gradient of bias from lm head (dbias lm head)
     *
     * @param dbias_lm_head_pred location device
     */
    float *dbias_lm_head_pred;

    float *bias_lm_head;

    float *WoT;

    /**
     * @brief Upstream gradient form the lm head linear layer.
     * @param dl_dh_output location device
     */
    float *dl_dh_output;

    float *d_add_residual_output;

    float *d_embedding;

    float *d_bias_q;
    float *d_bias_k;
    float *d_bias_v;

    float *d_weight_q;
    float *d_weight_k;
    float *d_weight_v;

    float *d_weight_output_project;
};