#pragma once
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "../include/utils.hpp"
#include "../include/linear.hpp"
#include "../include/linear.hpp"
#include "../forward/linear.hpp"
#include "../forward/embeddings.hpp"
#include "../forward/layer_norm.hpp"
#include "../include/attention_params.hpp"

extern "C" void vectorKernel(float *A, float *B, float *C, int N);
extern "C" void ScalerDvisionElem(float *arr, int batch, int n_head, int seq_len, int head_dim);
extern "C" void softmax(float *arr, float *out, int N, int seq_len, int n_head, int batch_size);
extern "C" void QKmatmul(float *Q, float *Kt, float *out, int M, int N, int batch_size, int num_heads);
extern "C" void UpperTriangularMasking(float *arr, float val, int batch_size, int n_head, int seq_len);
extern "C" void QKVMatmulFinal(float *QK, float *V, float *out, int seq_len, int d_head, int n_head, int batch_size);
extern "C" void ReformShapeWapper(float *arr, float *out, int batch_size, int seq_len, int d_model, int num_head, int head_dim);
extern "C" void ReformBNTH_BTC(float *arr, float *out, int batch_size, int seq_len, int d_model, int num_head, int head_dim);
extern "C" void SwapNS(float *arr, float *out, int batch_size, int seq_len, int num_heads, int head_dim, bool reverse);
class Attention
{
public:
    int d_model;
    int vocab_size;
    int num_heads;
    int seq_len;
    int batch_size;
    int head_dim;

    std::unique_ptr<Utility> utils = std::make_unique<Utility>();
    std::unique_ptr<Embeddings> embeddings; // called in the constructor good.
    std::unique_ptr<Linear> key;
    std::unique_ptr<Linear> query;
    std::unique_ptr<Linear> value;
    std::unique_ptr<Linear> outputProj;
    std::unique_ptr<LayerNorm> layerNorm;

    float *DeviceKt;
    float *DeviceQ;
    float *DeviceQKT;
    // shape is like B, num head, head_dim, seq_len
    float *BTC_MULTI_HEAD_BUFFER_HOST = nullptr; // most of this null pointer is uncessary but ok does not hurt.

    float *B_NUMHEAD_T_T; // QK^T host

    float *B_NUMHEAD_SEQLEN_HEADDIM;

    bool debug = true;

    float *deviceQKTSqrtD;

    float *deviceSoftmaxOut;

    // QKV part allocation, Softmax(QK/sqrt(d_model)) V
    float *QKVOutDeviceOut;
    float *deviceValue;

    // Allocation for re-connecting the heads.
    float *BTCHost = nullptr;
    float *BTCdevice;
    // same buffer for temp

    float *resedualOutDevice;

    float *tempX;
    float *x_device;

    // ----- Paramater required for standard attention backpropagation not the flash ------
    // at the learning stage of mine this is fine as well

    // UNTIL AND UNLESS YOU REALLY KNOW WHAT YOU ARE DOING
    // YOU WONT EVER KNOW WHAT WENT WRONG
    // BUGS IN KERNEL ARE THAT QUIET ON TOP OF THIS INSANE
    // MATHAMATICS

    float *softmaxed_qkv;

    float *S = nullptr;
    float *P = nullptr;
    float *O = nullptr;
    float *value_mat;

    float *Q_cache;
    float *K_cache;

    float *normalized_xt;
    // after the normalization we feed that "x" into the QK,andV paramater.

    float *doutput_bias;

    // for backpropagation we need to cache this and transpose of x
    // that is needed for the local gradients of the weights.
    float *output_porjection_x;
    float *output_project_xt;

    float *after_embedding_xt;

    float *normalized_x;

    float *layer_norm_x;

    Attention(
        int d_model,
        int vocab_size,
        int num_heads,
        int seq_len,
        int batch_size,
        bool debug)
    {

        this->d_model = d_model;
        this->vocab_size = vocab_size;
        this->num_heads = num_heads;
        this->seq_len = seq_len;
        this->batch_size = batch_size;
        this->debug = debug;

        // Its okay to free this in destructor because we are using this buffer throught the duration of
        // this object or iteration anyway.

        this->embeddings = std::make_unique<Embeddings>(
            this->d_model,
            this->vocab_size,
            this->seq_len,
            this->batch_size,
            debug);

        // Lets seed Q,K,V
        key = std::make_unique<Linear>(d_model, d_model, seq_len, batch_size, num_heads, debug);
        query = std::make_unique<Linear>(d_model, d_model, seq_len, batch_size, num_heads, debug);
        value = std::make_unique<Linear>(d_model, d_model, seq_len, batch_size, num_heads, debug);

        outputProj = std::make_unique<Linear>(d_model, d_model, seq_len, batch_size, num_heads, debug);

        layerNorm = std::make_unique<LayerNorm>(batch_size, seq_len, d_model);

        // Just to test and keep track of things
        // std::cout << "d_model: " << d_model << std::endl;
        // std::cout << "vocab_size: " << vocab_size << std::endl;
        // std::cout << "seq_len: " << seq_len << std::endl;
        // std::cout << "num_heads: " << num_heads << std::endl;
        // std::cout << "batch size: " << batch_size << std::endl;

        this->head_dim = d_model / num_heads;

        // --------- Buffer shape allocations -------------

        // Q K^t matmul device and host allocation
        cudaMalloc((void **)&DeviceKt, batch_size * num_heads * head_dim * seq_len * sizeof(float));
        cudaMalloc((void **)&DeviceQ, batch_size * num_heads * head_dim * seq_len * sizeof(float));
        cudaMalloc((void **)&DeviceQKT, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        // Q K^T Host
        BTC_MULTI_HEAD_BUFFER_HOST = (float *)malloc(batch_size * num_heads * head_dim * seq_len * sizeof(float));

        cudaMalloc((void **)&deviceQKTSqrtD, batch_size * num_heads * seq_len * seq_len * sizeof(float));
        cudaMalloc((void **)&softmaxed_qkv, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        B_NUMHEAD_T_T = (float *)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float)); // T,T end shape after QK^T

        // for softmax's
        // I do not deserve an internship so what, people who are doing this wont understand this so I am writing c++ to scare people off.
        cudaMalloc((void **)&deviceSoftmaxOut, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        cudaMalloc((void **)&QKVOutDeviceOut, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        cudaMalloc((void **)&deviceValue, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        B_NUMHEAD_SEQLEN_HEADDIM = (float *)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));

        // Here we will start the memory allocation for (B,T,C) since after computation the shapes are brought back
        this->BTCHost = (float *)malloc(batch_size * seq_len * d_model * sizeof(float)); // we aready have the in buffer allocation so I wont worry about that for right now., this is for output. we bring back the original shape.
        cudaMalloc((void **)&BTCdevice, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&tempX, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&resedualOutDevice, batch_size * seq_len * d_model * sizeof(float));

        // ---- S, P, O ---- are required for backpropagation allocate here according to reqired size.
        S = (float *)malloc(seq_len * batch_size * num_heads * head_dim * sizeof(float));

        cudaMalloc((void **)&P, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        O = (float *)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));

        // (B, n_head, T, head_dim) s
        cudaMalloc((void **)&value_mat, batch_size * num_heads * seq_len * head_dim * sizeof(float));
        cudaMalloc((void **)&Q_cache, batch_size * num_heads * seq_len * head_dim * sizeof(float));
        cudaMalloc((void **)&K_cache, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        cudaMalloc((void **)&doutput_bias, d_model * sizeof(float));
        cudaMalloc((void **)&x_device, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&output_porjection_x, batch_size * seq_len * d_model * sizeof(float));
        cudaMalloc((void **)&output_project_xt, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&normalized_xt, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&normalized_x, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&layer_norm_x, batch_size * seq_len * d_model * sizeof(float));
    };

    ~Attention()
    {

        cudaFree(DeviceKt);
        cudaFree(DeviceQ);
        cudaFree(DeviceQKT);

        cudaFree(softmaxed_qkv);

        cudaFree(deviceQKTSqrtD);

        (BTC_MULTI_HEAD_BUFFER_HOST != nullptr ? free(BTC_MULTI_HEAD_BUFFER_HOST) : void());

        (BTCHost != nullptr ? free(BTCHost) : void());

        cudaFree(tempX);

        cudaFree(deviceSoftmaxOut);
        cudaFree(deviceValue);
        cudaFree(QKVOutDeviceOut); // we could free in the individual method but if we never call them then its never free.

        cudaFree(BTCdevice);
        cudaFree(resedualOutDevice);

        cudaFree(doutput_bias);

        free(B_NUMHEAD_T_T);
        free(B_NUMHEAD_SEQLEN_HEADDIM);

        free(S);
        cudaFree(P);
        free(O);
        cudaFree(value_mat);

        cudaFree(output_porjection_x);
        cudaFree(output_project_xt);

        cudaFree(normalized_xt);
        cudaFree(normalized_x);

        cudaFree(Q_cache);
        cudaFree(K_cache);

        cudaFree(layer_norm_x);
    }

private:
    void cacheInputForOutputProj(float *expandedPointer, float *compactCache)
    {
        ReformBNTH_BTC(
            expandedPointer,
            compactCache,
            batch_size,
            seq_len,
            d_model,
            num_heads,
            head_dim);
    }

    void releaseBin(float *key, std::string name, int n)
    {
        float *h_key = (float *)malloc(n * sizeof(float));

        cudaMemcpy(h_key, key, n * sizeof(float), cudaMemcpyDeviceToHost);

        bulkRelease<float>(
            {{h_key, n * sizeof(float), name}});

        free(h_key);
    }

    void originalMultiHeadedQKVShapeDebugger(float *Q, float *K, float *V)
    {
        releaseBin(Q, "q_org.bin", batch_size * seq_len * num_heads * head_dim);
        releaseBin(K, "k_org.bin", batch_size * seq_len * num_heads * head_dim);
        releaseBin(V, "v_org.bin", batch_size * seq_len * num_heads * head_dim);
    }

    void resehapedQKVShapeDebugger(float *Q, float *K, float *V)
    {
        releaseBin(Q, "q_reshaped.bin", batch_size * seq_len * num_heads * head_dim);
        releaseBin(K, "k_reshaped.bin", batch_size * seq_len * num_heads * head_dim);
        releaseBin(V, "v_reshaped.bin", batch_size * seq_len * num_heads * head_dim);
    }

public:
    // This method divides by sqrt of d_model so that atlast forward lgoic in this program looks readable to a normal human being.
    void scalerDvisionAcrossMat(float *QKt)
    {
        ScalerDvisionElem(
            QKt,
            batch_size,
            num_heads,
            seq_len,
            head_dim);
    }

    void masking(float *arr, float val)
    {
        UpperTriangularMasking(arr, val, batch_size, num_heads, seq_len);
    }

    // BEFORE (B, T, n_head, head_dim) AFTER: (B, T, C)
    float *ReformShape(float *arr)
    {
        ReformShapeWapper(
            arr,
            BTCdevice,
            batch_size,
            seq_len,
            d_model,
            num_heads,
            head_dim);

        return BTCdevice;
    }

    float *forward(Batch currentBatch)
    {
        float *x = embeddings->forward(currentBatch);

        cudaMemcpy(tempX, x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);

        if (debug)
            releaseBin(tempX, "temp_x.bin", batch_size * seq_len * d_model);

        // Layer normalization of x. Note: that the x pointer is modified.
        layerNorm->forward(x);

        cudaMemcpy(layer_norm_x, x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);

        if (debug)
        {
            releaseBin(x, "layer_norm_forward.bin", batch_size * seq_len * d_model);
            releaseBin(layerNorm->getGammaDevice(), "gamma_device.bin", d_model);
            releaseBin(layerNorm->getBeta(), "beta_device.bin", d_model);
        }

        // Linear transformation of QKV matrix
        float *Q = query->forward(x);
        float *K = key->forward(x);
        float *V = value->forward(x);

        // (B, T, C) to (batch_size, seq_len, num_head, head_dim)
        Q = query->reshapeHead();
        K = key->reshapeHead();
        V = value->reshapeHead();

        // release one weight to check what's happenning here.
        // weights are generated equally right

        if (debug)
        {
            releaseBin(query->getWeightDevice(), "attn_weight_q.bin", d_model * d_model);
            originalMultiHeadedQKVShapeDebugger(Q, K, V);
        }

        // (batch_size, seq_len, num_head, head_dim) to (batch_size, num_head, seq_len, head_dim)

        Q = query->transposeMiddle4DTensor(); // (batch_size, num_head, seq_len, head_dim)
        K = key->transposeMiddle4DTensor();   // (batch_size, num_head, seq_len, head_dim)
        V = value->transposeMiddle4DTensor(); // (batch_size, num_head, seq_len, head_dim)

        // QKV cache
        cudaMemcpy(Q_cache, Q, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(K_cache, K, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(value_mat, V, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        if (debug)
            resehapedQKVShapeDebugger(Q, K, V);

        // (batch_size, num_head, seq_len, head_dim) to (batch_size, num_head, head_dim, seq_len)
        key->teansposeKeyForAttnScore(DeviceKt);

        if (debug)
            releaseBin(DeviceKt, "tranposed_k.bin", batch_size * seq_len * num_heads * head_dim);

        QKmatmul(Q, DeviceKt, DeviceQKT, seq_len, head_dim, batch_size, num_heads);

        if (debug)
            releaseBin(DeviceQKT, "qkt.bin", batch_size * seq_len * num_heads * seq_len);

        scalerDvisionAcrossMat(DeviceQKT); // sqrt(head_dim)

        if (debug)
            releaseBin(DeviceQKT, "sqrt_d_head.bin", batch_size * seq_len * num_heads * seq_len);

        // -1e9f
        masking(DeviceQKT, -INFINITY);

        if (debug)
            releaseBin(DeviceQKT, "casual_masking.bin", batch_size * seq_len * num_heads * seq_len);

        softmax(DeviceQKT, deviceSoftmaxOut, batch_size * num_heads * seq_len * seq_len, seq_len, num_heads, batch_size);

        // cache P = softmax(x)
        //     cudaMemcpy(P, deviceSoftmaxOut, batch_size * seq_len * num_heads * seq_len * sizeof(float), cudaMemcpyDeviceToDevice);

        if (debug)
            releaseBin(deviceSoftmaxOut, "softmaxed_attn.bin", batch_size * seq_len * num_heads * seq_len);

        QKVMatmulFinal(
            deviceSoftmaxOut, // (batch_size, num_head, seq_len, seq_len)
            V,                // (batch_size, num_head, seq_len, head_dim)
            softmaxed_qkv,    // (batch_size, num_head, seq_len, head_dim)
            seq_len,
            head_dim,
            num_heads,
            batch_size);

        if (debug)
            releaseBin(softmaxed_qkv, "attn_score.bin", batch_size * num_heads * seq_len * head_dim);

        float *reformed_tensor = ReformShape(softmaxed_qkv);

        if (debug)
            releaseBin(reformed_tensor, "re_shaped_multi_head.bin", batch_size * seq_len * d_model);

        //    cudaMemcpy(output_porjection_x, reformed_tensor, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);

        float *projectedBTC = outputProj->forward(reformed_tensor);

        float *residual_out = addResidual(projectedBTC, tempX);

        if (debug)
            releaseBin(residual_out, "final_attn_out.bin", batch_size * seq_len * d_model);
        if (debug)
            releaseBin(projectedBTC, "out_put_proj.bin", batch_size * seq_len * d_model);

        debug = false;

        return residual_out;
    };

    float *addResidual(float *input, float *temp)
    {

        vectorKernel(input, temp, resedualOutDevice, batch_size * seq_len * d_model);

        return resedualOutDevice;
    }

    AttentionParamaters attentionParamaters; // just the reference so stack allocation is fine

    AttentionParamaters getParamaters()
    {

        return {
            S, // actual value copied from QKV
            P,
            O,
            value_mat,

            BTCdevice,
            softmaxed_qkv,

            Q_cache,
            K_cache,

            tempX, // last pointer my guess is this wont be modified anytime soon.
            normalized_xt,
            layer_norm_x,

            layerNorm->getMean(),
            layerNorm->getStdDev(),
            layerNorm->getGammaDevice(),
            layerNorm->getBeta(),

            query->getWeightDevice(),
            key->getWeightDevice(),
            value->getWeightDevice(),

            outputProj->getWeightDevice(),
            outputProj->getBaiasDevice(),

            doutput_bias,
            embeddings->deviceX,

            query->getBaiasDevice(),
            key->getBaiasDevice(),
            value->getBaiasDevice(),

            output_porjection_x,
            output_project_xt,

            embeddings->getEmbeddingParmaterDevice()};
    }
};