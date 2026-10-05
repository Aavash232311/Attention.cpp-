#pragma once
#include "../include/helper.hpp"
#include "../include/utils.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./interface_back.hpp"

using namespace std;

extern "C" void wt_upstream(float *w, float *wt, int a, int b);
extern "C" void dbias(float *G, float *dbias, int B, int T, int C, bool sync);
extern "C" void dl_dh_upstream(float *A, float *B, float *C, int a, int b, int c, int d, bool sync);
extern "C" void ReformBNTH_BTC(float *arr, float *out, int batch_size, int seq_len, int d_model, int num_head, int head_dim);
extern "C" void addThreeTensor(float *A, float *B, float *C, float *Out, int batch_size, int seq_len, int d_model);
extern "C" void layernorm_backward(float *x, float *G, float *mc, float *sdc, float *gamma, float *dgamma, float *dbeta, int B, int T, int C);
extern "C" void addTwoTensor(float *A, float *B, float *Out, int batch_size, int seq_len, int d_model);
extern "C" void updateTokenEmbedding(float *G, float *d_emebdding, int *token_ids, int batch_size, int seq_len, int d_model, int vocab_size);
extern "C" void dl_dw_upstream(float *deta, float *ht, float *out, int B, int T, int C, int vocab_size);
extern "C" void lm_head_transpose_h(float *h, float *out, int B, int T, int C);
class FlashAttentionLinear : virtual public AutoGradEngine
{

private:
    /**
     * @brief calculates the gradeint along QKV for all the weight and biases.
     * @bug         Note: ISSUE
        dQ, dK, dV are each upstream gradient for
        dweight and dbias for QKV after that we sum the weights
        for the layer norm. And thats how the chain gets completed
     */
    void linearBackForQKV()
    {
        // Note:- These task does not have to be sequential but we have that sync call in that wrapper we will let it be as it is.
        // weight backpropagation for Q

        // transpose x that goes into the all
        // three q,k,and v linear.

        lm_head_transpose_h(
            model_paramaters.attention_head.normalized_x,
            model_paramaters.attention_head.normalized_xt,
            batch_size,
            seq_len,
            d_model);

        // for each of the linear layer we need to get the
        // backpropagation done.

        linearBack->backward(
            model_paramaters.attention_head.normalized_x,
            model_paramaters.qUp,
            model_paramaters.attention_head.device_WQ,
            model_paramaters.WqT,
            model_paramaters.attention_head.normalized_xt,
            model_paramaters.d_weight_q,
            model_paramaters.d_bias_q,
            model_paramaters.dqWt,
            false,
            false);

        linearBack->backward(
            model_paramaters.attention_head.normalized_x,
            model_paramaters.kUp,
            model_paramaters.attention_head.device_WK,
            model_paramaters.WkT,
            model_paramaters.attention_head.normalized_xt,
            model_paramaters.d_weight_k,
            model_paramaters.d_bias_k,
            model_paramaters.dkWt,
            false,
            false);

        linearBack->backward(
            model_paramaters.attention_head.normalized_x,
            model_paramaters.vUp,
            model_paramaters.attention_head.device_WV,
            model_paramaters.WvT,
            model_paramaters.attention_head.normalized_xt,
            model_paramaters.d_weight_v,
            model_paramaters.d_bias_v,
            model_paramaters.dvWt,
            false,
            false);

        if (debug)
        {
            // float *x = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));
            // float *xt = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));

            // cudaMemcpy(x, model_paramaters.attention_head.normalized_x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(xt, model_paramaters.attention_head.normalized_xt, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

            // cout << "Normalized x" << endl;
            // this->utils->printFlatArray3D(x, batch_size, seq_len, d_model);

            // cout << "Transposed x" << endl;
            // this->utils->printFlatArray3D(xt, batch_size, d_model, seq_len);

            // free(xt);
            // free(x);
        }

        if (debug)
            pyDebuggerReleaseStage11();
    }

    void copyWeightQKVtoDevice()
    {
        // from attention pointer has the thing inside of CPU
        // copy them all to GPU

        // K and V are allocated elsewhere just re-using this pointer

        // cudaMemcpy(model_paramaters.Wk, model_paramaters.attention_head.host_WK, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(model_paramaters.WQ, model_paramaters.attention_head.host_WQ, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(model_paramaters.WV, model_paramaters.attention_head.host_WV, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);

        // reanspose shape (C, C) -> (C, C) output variabel WKT, WqT, WvT
        wt_upstream(model_paramaters.attention_head.device_WK, model_paramaters.WkT, d_model, d_model);
        wt_upstream(model_paramaters.attention_head.device_WQ, model_paramaters.WqT, d_model, d_model);
        wt_upstream(model_paramaters.attention_head.device_WV, model_paramaters.WvT, d_model, d_model); // out shape (d_mdoel, d_model)

        // recalling the shape here

        // dQ = 1/sqrt(dk) G K      shape=(batch_size, num_heads, seq_len, head_dim)
        // dK = 1/sqrt(dk) G^T Q    shape=(batch_size, num_heads, seq_len, head_dim)
        // dV =  P^T G              shape=(batch_size, num_heads, seq_len, head_dim)

        // dQ, dK, dV shape of (batch_size, num_head, seq_len, head_dim) to (batch_size, seq_len, d_model) output vUp, qUp, kUp
        ReformBNTH_BTC(model_paramaters.dV, model_paramaters.vUp, batch_size, seq_len, d_model, num_heads, head_dim);
        ReformBNTH_BTC(model_paramaters.dQ, model_paramaters.qUp, batch_size, seq_len, d_model, num_heads, head_dim);
        ReformBNTH_BTC(model_paramaters.dK, model_paramaters.kUp, batch_size, seq_len, d_model, num_heads, head_dim);

        linearBackForQKV();

        // Problem with this matmul kernel but I will look at it, its been a rough week

        // matirx multiplication, shape BTC, CC, CC
        dl_dh_upstream(model_paramaters.qUp, model_paramaters.WqT, model_paramaters.dqWt, batch_size, seq_len, d_model, d_model, true);
        dl_dh_upstream(model_paramaters.kUp, model_paramaters.WkT, model_paramaters.dkWt, batch_size, seq_len, d_model, d_model, true);

        dl_dh_upstream(model_paramaters.vUp, model_paramaters.WvT, model_paramaters.dvWt, batch_size, seq_len, d_model, d_model, true);

        // pass in the compact shape (B,T,C)
        addThreeTensor(model_paramaters.dqWt, model_paramaters.dkWt, model_paramaters.dvWt, model_paramaters.G_x_hat, batch_size, seq_len, d_model);

        // copy that x after the net_embedding to device

        cudaMemcpy(model_paramaters.attention_head.x,
                   model_paramaters.attention_head.x,
                   batch_size * seq_len * d_model * sizeof(float),
                   cudaMemcpyHostToDevice);

        /***
         * @warning the first arguement pointer is modified here.
         * Which might cause confusion and problems but it is what it is.
         *
         */
        layernorm_backward(
            model_paramaters.attention_head.x,
            model_paramaters.G_x_hat,
            model_paramaters.attention_head.mean_cache,
            model_paramaters.attention_head.std_dev_cache,
            model_paramaters.attention_head.d_gamma,
            model_paramaters.dgamma,
            model_paramaters.debeta,
            batch_size,
            seq_len,
            d_model);

        if (debug)
            pyDebuggerReleaseStage8();

        // my anxiety and lack of sleep has caused be to do many simple mistakes recently :)
        addTwoTensor(
            model_paramaters.dl_dh_output,
            model_paramaters.attention_head.x,
            model_paramaters.d_add_residual_output,
            batch_size,
            seq_len,
            d_model);

        // update the embedding gradient
        updateTokenEmbedding(
            model_paramaters.attention_head.x,
            model_paramaters.d_embedding,
            model_paramaters.attention_head.token_id,
            batch_size,
            seq_len,
            d_model,
            vocab_size);

        if (debug)
            pyDebuggerReleaseStage10();
    }

public:
    FlashAttentionLinear(int d_model, int vocab_size, int num_heads,
                         int seq_len, int batch_size, bool debug)
        : AutoGradEngine(d_model, vocab_size, num_heads, seq_len, batch_size, debug)
    {
    }

    /*
        Shape note:

        Shape dK: (batch_size, num_heads, seq_len, head_dim)
        Shape dQ: (batch_size, num_heads, seq_len, head_dim)
        Shape dV: (batch_size, num_heads, seq_len, head_dim)

        Shape W's   : (d_model, d_model)
        Shape W'ts  : (d_model, d_model)

        We aready have the transpose kernel here.
        Not sure to re-use that kerenl or write a new one reading in a transpose way, too old for that.


    */

    void weightTransposeAttn()
    {
        this->copyWeightQKVtoDevice();
    }

    void NormLinearNet()
    {
        this->weightTransposeAttn();
    }
};