#pragma once
#include "../include/helper.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>


#include "../include/tensor.hpp"
#include "../include/utils.hpp"
#include "../forward/linear.hpp"
#include "../include/linear.hpp"
#include "../include/p_head.hpp"
#include "../optimizer/AdamW.hpp"
#include "../include/cache_in.hpp"
#include "../forward/embeddings.hpp"
#include "../include/data_loader.hpp"
#include "../include/netattention.hpp"
#include "../forward/attention_head.hpp"
#include "../optimizer/ds/AdamW_config.hpp"
#include "../optimizer/optimizer_debugger.hpp"
#include "../backpropagation/interface_back.hpp"
#include "../backpropagation/debugger_release.hpp"

using namespace std;

extern "C" void softmax2D(float *arr, float *out, int batch_size, int seq_len, int vocab_size);
extern "C" void CrossEntropy(float *x, int *y, float *oneHotOut, float *lossOut, int batch_size, int seq_len, int vocab_size);

class AttentionInterface
{
public:
    int d_model;
    int num_heads;
    int vocab_size;
    int batch_size;
    int seq_len;
    int head_dim;
    bool runDebugger = false;
    bool drop_last = true;

    const std::vector<Batch> &data;
    std::unique_ptr<Attention> attention;
    std::unique_ptr<DataLoader> dataLoader;
    std::unique_ptr<Utility> utils;
    std::unique_ptr<OptimizerDebugger> optimizer;
    std::unique_ptr<AdamWMemConfig> adamMemConfig;
    std::unique_ptr<Helper> helper;

    // turn those result into probability score
    std::unique_ptr<Linear> lm_head;

    bool debug = true;

    // DEVICE SOFTMAX BEFORE CROSS ENTROPY LOSS IN
    std::unique_ptr<Tensor<float>> DeviceSoftmaxBLin;
    // DEVICE SOFTMAX BEFORE CROSS ENTROPY LOSS OUT
    std::unique_ptr<Tensor<float>> DeviceSoftmaxBLout;

    // ALLOCATE MEMORY FOR Y
    std::unique_ptr<Tensor<float>> yHotEncodeDeviceOut;
    std::unique_ptr<Tensor<float>> outCrossEntropyDevice;
    std::unique_ptr<Tensor<float>> outCrossEntropyHost;
    std::unique_ptr<AutogradEngineDebuggerRelease> autograd;

    // LOW LEVEL BY DESIGN IS LITTLE BIT MESSY COMES WITH A TRADE OFFS ANYWAY

    std::unique_ptr<Tensor<float>> upstream_softmax_ce;
    std::unique_ptr<Tensor<float>> dl_dw_out_device;
    std::unique_ptr<Tensor<float>> d_softmax_upstream;

    // GPU buffer for the autograd engine in the attention interface.
    FlashAttentionPointers modelParameters{};

    // ---------- Autograd engine declaration --------------------
    std::unique_ptr<Tensor<float>> ht_device;

    // ---------- Autograd engine weight lm head ---------------------
    std::unique_ptr<Tensor<float>> w_out_d;

    // For the net gradient we also want to allocate for
    // WkT and WvT

    std::unique_ptr<Tensor<float>> WkT;
    std::unique_ptr<Tensor<float>> WvT;
    std::unique_ptr<Tensor<float>> WqT;

    std::unique_ptr<Tensor<float>> Wk;
    std::unique_ptr<Tensor<float>> WV;
    std::unique_ptr<Tensor<float>> WQ;

    // ------------ Allocation has attention class ---------------------
    std::unique_ptr<Tensor<float>> S_device;
    std::unique_ptr<Tensor<float>> P_device;
    std::unique_ptr<Tensor<float>> O_device;
    // P^T and V^T device allocation
    std::unique_ptr<Tensor<float>> P_T_device_out;
    std::unique_ptr<Tensor<float>> V_T_device_out;
    std::unique_ptr<Tensor<float>> Un_contact_G_Upstream;
    std::unique_ptr<Tensor<float>> Contact_G_Upstream;
    std::unique_ptr<Tensor<float>> dV; // O = PV one of the derivative terms.
    std::unique_ptr<Tensor<float>> dP; // same derivative terms.
    std::unique_ptr<Tensor<float>> dQ;
    std::unique_ptr<Tensor<float>> dK;
    std::unique_ptr<Tensor<float>> ppt;
    std::unique_ptr<Tensor<float>> d_score_t;
    std::unique_ptr<Tensor<float>> qUp;
    std::unique_ptr<Tensor<float>> kUp;
    std::unique_ptr<Tensor<float>> vUp;
    std::unique_ptr<Tensor<float>> dqWt;
    std::unique_ptr<Tensor<float>> dkWt;
    std::unique_ptr<Tensor<float>> dvWt;
    std::unique_ptr<Tensor<float>> G_x_hat; // total added up tensor
    std::unique_ptr<Tensor<float>> delta_beta;
    std::unique_ptr<Tensor<float>> delta_gamma;
    std::unique_ptr<Tensor<float>> WoT;
    std::unique_ptr<Tensor<float>> dl_dh_output;
    std::unique_ptr<Tensor<float>> dbias_lm_head_pred;
    std::unique_ptr<Tensor<float>> d_add_residual_output;
    std::unique_ptr<Tensor<float>> d_embedding;
    std::unique_ptr<Tensor<float>> d_bias_q;
    std::unique_ptr<Tensor<float>> d_bias_k;
    std::unique_ptr<Tensor<float>> d_bias_v;
    std::unique_ptr<Tensor<float>> d_weight_q;
    std::unique_ptr<Tensor<float>> d_weight_k;
    std::unique_ptr<Tensor<float>> d_weight_v;
    std::unique_ptr<Tensor<float>> d_weight_output_project;

    std::unique_ptr<Tensor<float>> dx;
    bool engine_run;
    std::unique_ptr<Tensor<float>> attn_head_out;
    AttentionInterface(
        int d_model,
        int num_heads,
        int batch_size,
        int seq_len,
        bool drop_last,
        int vocab_size,
        const std::vector<Batch> &data,
        bool debug,
        bool engine_run = false);

    [[nodiscard]] LinearParams getLmHeadParams() const {return LinearParams{.Weight = lm_head->getWeightDevice(), .Bias = lm_head->getBiasDevice()};}
    void softmaxAcrossProbabilityCrossEntropyLoss(float *probability, int *y) const;

    // note lookup contains seq_len = 12 for example whatever its trained on
    // and we are trying to retrieve something different from forward pass so
    // we have a different seq_len for this forward pass called "prompt_seq_len"
    void forward(std::string input, std::vector<char> &charPool, int gen = 128);

    void train(int epoch);
};