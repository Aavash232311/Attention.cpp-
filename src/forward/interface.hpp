#pragma once
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <algorithm>

#include "../include/tensor.hpp"
#include "../include/utils.hpp"
#include "../forward/linear.hpp"
#include "../include/linear.hpp"
#include "../include/p_head.hpp"
#include "../optimizer/AdamW.hpp"
#include "../include/cache_in.hpp"
#include "../include/cache_out.hpp"
#include "../forward/layer_norm.hpp"
#include "../forward/embeddings.hpp"
#include "../include/data_loader.hpp"
#include "../include/netattention.hpp"
#include "../forward/attention_head.hpp"
#include "../optimizer/ds/AdamW_config.hpp"
#include "../optimizer/optimizer_debugger.hpp"
#include "../backpropagation/interface_back.hpp"
#include "../backpropagation/flash_attention.hpp"
#include "../backpropagation/debugger_release.hpp"

using namespace std;

extern "C" void softmax2D(float *arr, float *out, int batch_size, int seq_len, int vocab_size);
extern "C" void CrossEntropy(float *x, int *y, float *oneHotOut, float *lossOut, int batch_size, int seq_len, int vocab_size);

class AttentionInterface
{

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

    // turn those result into proballity score
    std::unique_ptr<Linear> lm_head;

    bool debug = true;

    // DEVICE SFOTMAX BEFORE CROSS ENTROPY LOSS IN
    std::unique_ptr<Tensor<float>> DeviceSoftmaxBLin;
    // DEVICE SOFTMAX BEFORE CORSS ENTROPY LOSS OUT
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
    FlashAttentionPointers modelParamaters;

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

    // ------------ Allocation fhas attention class ---------------------

    std::unique_ptr<Tensor<float>> S_device;
    std::unique_ptr<Tensor<float>> P_device;
    std::unique_ptr<Tensor<float>> O_device;

    // P^T and V^T device alloication

    std::unique_ptr<Tensor<float>> P_T_device_out;
    std::unique_ptr<Tensor<float>> V_T_device_out;

    std::unique_ptr<Tensor<float>> Uncontact_G_Upstream;
    std::unique_ptr<Tensor<float>> Contact_G_Upstream;

    std::unique_ptr<Tensor<float>> dV; // O = PV one of the derivative terms.
    std::unique_ptr<Tensor<float>> dP; // same derviative terms.

    std::unique_ptr<Tensor<float>> dQ;
    std::unique_ptr<Tensor<float>> dK;

    // softmax activation derivative terms.

    std::unique_ptr<Tensor<float>> ppt;

    std::unique_ptr<Tensor<float>> d_score_t;

    /*
        Net G for layer norm very end formula in
        flashback.md
    */

    std::unique_ptr<Tensor<float>> qUp;
    std::unique_ptr<Tensor<float>> kUp;
    std::unique_ptr<Tensor<float>> vUp;

    std::unique_ptr<Tensor<float>> dqWt;
    std::unique_ptr<Tensor<float>> dkWt;
    std::unique_ptr<Tensor<float>> dvWt;

    std::unique_ptr<Tensor<float>> G_x_hat; // total added up tensor

    std::unique_ptr<Tensor<float>> dbeta;
    std::unique_ptr<Tensor<float>> dgamma;

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

private:
    // ----------- TEMPORARY DEBUGGER SCRIPT ---------------------

    // B,T,C shape use if you want to see and inspeace device
    void DebugBTCFlatArray3D(
        float *d_arr,
        int B,
        int T,
        int C // vocab size
    )
    {
        float *h_arr = (float *)malloc(B * T * C * sizeof(float));

        cudaMemcpy(h_arr, d_arr, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

        utils->printLastOneOf3D(h_arr, B, T, C);

        free(h_arr);
    }

public:
    AttentionInterface(
        int d_model,
        int num_heads,
        int batch_size,
        int seq_len,
        bool drop_last,
        int vocab_size,
        const std::vector<Batch> &data,
        bool debug,
        bool engine_run = false) : data(data)
    {
        this->d_model = d_model;
        this->vocab_size = vocab_size;
        this->batch_size = batch_size;
        this->seq_len = seq_len;
        this->drop_last = drop_last;
        this->num_heads = num_heads;
        this->debug = debug;
        this->engine_run = engine_run;

        this->head_dim = d_model / num_heads;

        utils = std::make_unique<Utility>();

        attention = std::make_unique<Attention>(
            d_model,
            vocab_size,
            num_heads,
            seq_len,
            batch_size,
            debug);

        optimizer = std::make_unique<OptimizerDebugger>(3e-3f, 0.0f, 0.9f, 0.98f, 1e-8f, debug);

        // pass in the derived class for proper inheritance
        autograd = std::make_unique<AutogradEngineDebuggerRelease>(
            d_model,
            vocab_size,
            num_heads,
            seq_len,
            batch_size,
            debug,
            engine_run);

        adamMemConfig = std::make_unique<AdamWMemConfig>(
            batch_size,
            seq_len,
            d_model,
            vocab_size,
            num_heads);

        // BUFFER CPU/GPU allocation for the auto grad engine
        upstream_softmax_ce = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, vocab_size}, DEVICE);
        //   dataLoader = std::make_unique<DataLoader>(batch_size, data, seq_len, drop_last);

        lm_head = std::make_unique<Linear>(d_model, vocab_size, seq_len, batch_size, num_heads, debug);

        // wont something like this reserve GDDR RAM for too long till the liftspan of the object? Yes. -Avash
        DeviceSoftmaxBLin = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, vocab_size}, DEVICE);
        // Tensor<float> DeviceSoftmaxBLout(vector<int>{batch_size, seq_len, vocab_size}, DEVICE);
        DeviceSoftmaxBLout = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, vocab_size}, DEVICE);

        outCrossEntropyDevice = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len}, DEVICE); // (N) across all of the BT we will have the loss.
        cudaMemset(outCrossEntropyDevice->get(), 0, batch_size * seq_len * sizeof(float));        // make it zero for the things that we are not touching.

        // YOU DO NEED THIS PART BECAUSE ANOTHER KERNEL FOR THE CORSS ENTROPY LOSS WILL BE USING THIS.
        yHotEncodeDeviceOut = std::make_unique<Tensor<float>>(std::vector<int>{vocab_size, seq_len, batch_size}, DEVICE); // (B, T)
        cudaMemset(yHotEncodeDeviceOut->get(), 0, batch_size * seq_len * vocab_size * sizeof(float));        // make default 0

        outCrossEntropyHost = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, vocab_size}, HOST);

        ht_device = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);

        dl_dw_out_device = std::make_unique<Tensor<float>>(std::vector<int>{d_model, vocab_size}, DEVICE);

        // ------------ for w in lm head --------------

        w_out_d = std::make_unique<Tensor<float>>(std::vector<int>{d_model, vocab_size}, DEVICE);

        // ----- For flash attention kernel -------------------
        S_device = std::make_unique<Tensor<float>>(std::vector<int>{seq_len, batch_size, num_heads, head_dim}, DEVICE);
        P_device = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, seq_len}, DEVICE);
        O_device = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, head_dim}, DEVICE);

        // output, I know not the most efficient kernel or way but lets get to the end-result here first.
        P_T_device_out = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, seq_len}, DEVICE);
        V_T_device_out = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, head_dim}, DEVICE);

        Uncontact_G_Upstream = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, num_heads, head_dim}, DEVICE);

        Contact_G_Upstream = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);

        dV = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, head_dim}, DEVICE);
        dP = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, seq_len}, DEVICE);

        ppt = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, seq_len}, DEVICE);

        // dQ and dK for the QK^T backpropagation
        dQ = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, num_heads, head_dim}, DEVICE);
        dK = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, num_heads, head_dim}, DEVICE);

        d_score_t = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, seq_len}, DEVICE);

        WkT = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);
        WvT = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);
        WqT = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);

        Wk = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);
        WV = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);
        WQ = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);

        qUp = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);
        kUp = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);
        vUp = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);

        G_x_hat = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);

        dqWt = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);
        dkWt = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);
        dvWt = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);

        dbeta = std::make_unique<Tensor<float>>(std::vector<int>{d_model}, DEVICE);
        dgamma = std::make_unique<Tensor<float>>(std::vector<int>{d_model}, DEVICE);

        dbias_lm_head_pred = std::make_unique<Tensor<float>>(std::vector<int>{vocab_size}, DEVICE);

        WoT = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);

        dl_dh_output = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);

        d_add_residual_output = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);

        d_embedding = std::make_unique<Tensor<float>>(std::vector<int>{vocab_size, d_model}, DEVICE);

        d_bias_q = std::make_unique<Tensor<float>>(std::vector<int>{d_model}, DEVICE);
        d_bias_k = std::make_unique<Tensor<float>>(std::vector<int>{d_model}, DEVICE);
        d_bias_v = std::make_unique<Tensor<float>>(std::vector<int>{d_model}, DEVICE);

        d_weight_q = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);
        d_weight_k = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);
        d_weight_v = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);

        d_weight_output_project = std::make_unique<Tensor<float>>(std::vector<int>{d_model, d_model}, DEVICE);

        d_softmax_upstream = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, num_heads, seq_len, seq_len}, DEVICE);

        dx = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);

        attn_head_out = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, d_model}, DEVICE);
    }

    ~AttentionInterface()
    {
        // cudaFree(DeviceSoftmaxBLin);
        //  cudaFree(DeviceSoftmaxBLout);


        // free flash-attention device allocations

    }

    LinearParams getLmHeadParams()
    {
        return LinearParams{lm_head->getWeightDevice(), lm_head->getBaiasDevice()};
    }

    void softmaxAcrossProballityCrossEntropyLoss(float *probality, int *y)
    {

        // this accounts for whatever we are doing softmax on being greater than 32 i.e warp size.
        // THIS IS SOFTMAX ACROSS the logits (B, T, vocab_size)

        /*
            DeviceSoftmaxBLout: Proballity from logits (predicted)
            yHotEncodeDeviceOut: Actual Y (B, T) -> (B, T, vocab_size)
            outCrossEntropyDevice: CE out flat list
        */

        if (debug)
            attention->releaseBin(probality, "lm_head_out.bin", batch_size * seq_len * vocab_size);

        softmax2D(
            probality,
            DeviceSoftmaxBLout->get(),
            batch_size,
            seq_len,
            vocab_size);

        // utils->printFlatArray3D(DeviceSoftmaxBLout, batch_size, seq_len, vocab_size, false, "device");

        if (debug)
            attention->releaseBin(DeviceSoftmaxBLout->get(), "ce_softmax_out.bin", batch_size * seq_len * vocab_size);

        CrossEntropy(
            DeviceSoftmaxBLout->get(), // out from softmaxed
            y,
            yHotEncodeDeviceOut->get(),
            outCrossEntropyDevice->get(),
            batch_size,
            seq_len,
            vocab_size);

        cudaMemcpy(outCrossEntropyHost->get(), outCrossEntropyDevice->get(), batch_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        // utils->printFlatArray2D(outCrossEntropyDevice, batch_size, seq_len, "device");

        if (debug)
        {
            //  Here basically every element of the y must have its position encoded based on all of the vocab_size
            // std::cout << "Vocab size of all BT" << std::endl;
            // utils->printLastOneOf3D(probality, batch_size, seq_len, vocab_size);

            // std::cout << "Before one hot encode" << std::endl;
            // utils->printFlatArray2D(y, batch_size, seq_len);

            // // Shape
            // std::cout << " After one hot encode " << std::endl;
            // utils->printFlatArray3D(outHotEncodeOut, batch_size, seq_len, vocab_size);

            // std::cout << "Apply the cross entropy loss" << std::endl;
            // utils->printFlatArray1D(outCrossEntropyHost, batch_size * seq_len);

            // std::cout << "Predicted" << std::endl;
            // DebugBTCFlatArray3D(DeviceSoftmaxBLout, batch_size, seq_len, vocab_size);

            // std::cout << "Actual" << std::endl;
            // DebugBTCFlatArray3D(yHotEncodeDeviceOut, batch_size, seq_len, vocab_size);
        }
    }

    // note lookup contains seq_len = 12 for example whatever its trained on
    // and we are trying to retrive something different from forward pass so
    // we have a different seq_len for this forward pass called "prompt_seq_len"

    void forward(std::string input, std::vector<char> &charPool, int gen = 128)
    {
        helper = std::make_unique<Helper>(charPool);

        const int prompt_seq_len = seq_len;

        // Encode prompt
        vector<int> encodedInput = helper->encoder(input);
        if (encodedInput.size() < static_cast<size_t>(prompt_seq_len))
        {
            throw std::invalid_argument("Generation prompt must encode to at least " + std::to_string(prompt_seq_len) + " known characters.");
        }

        float *softmaxedProb;
        cudaMalloc(
            (void **)&softmaxedProb,
            prompt_seq_len * vocab_size * sizeof(float));

        float *softmaxedProbHost = (float *)malloc(
            prompt_seq_len * vocab_size * sizeof(float));

        for (int i = 0; i < gen; ++i)
        {
            vector<int> window(
                encodedInput.end() - prompt_seq_len,
                encodedInput.end());

            int *tokens = utils->vectorToFlatStrip(window, "gpu");

            float *emb = attention->embeddings->forward(
                tokens, 1, prompt_seq_len);

            float *attnScore = attention->forward(
                emb, 1, prompt_seq_len);

            float *logits = lm_head->forward(
                attnScore, 1, prompt_seq_len);

            softmax2D(
                logits,
                softmaxedProb,
                1,
                prompt_seq_len,
                vocab_size);

            cudaMemcpy(
                softmaxedProbHost,
                softmaxedProb,
                prompt_seq_len * vocab_size * sizeof(float),
                cudaMemcpyDeviceToHost);

            int offset = (prompt_seq_len - 1) * vocab_size;

            int index = 0;
            float maxProb = softmaxedProbHost[offset];

            for (int j = 1; j < vocab_size; ++j)
            {
                float currentProb = softmaxedProbHost[offset + j];

                if (currentProb > maxProb)
                {
                    maxProb = currentProb;
                    index = j;
                }
            }
            encodedInput.push_back(index);

            cudaFree(tokens);
        }

        auto decode = helper->decoder(encodedInput);

        for (int i = 0; i < decode.size(); ++i)
        {
            cout << decode[i];
        }

        cout << "\n";

        free(softmaxedProbHost);
        cudaFree(softmaxedProb);
    }

    void train(int epoch)
    {
        float loss = 0.0f;

        for (int i = 0; i < epoch; ++i)
        {
            float loss_per_batch = 0.0f;
            int batch_count = 0;

            for (int j = 0; j < data.size(); j++)
            {
                // std::cout << batch.width << std::endl;
                // utils->printFlatArray2D(batch.x, seq_len, batch.width);

                // Shape (B, T, C) only pointer dependent upon the channel dimension is here
                // And there is an illegal memory access somewhere here.

                Batch batch = data[j]; // current batch maybe in GPU/CPU

                float *x = attention->forward(batch);

                cudaMemcpy(attn_head_out->get(), x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);

                float *prob = lm_head->forward(x); // Shape (B, T, vocab_size) x is not changed here.

                // utils->printFlatArray2D(attention->key->getMultiHeadedDeivce(), d_model, d_model, "device");
                // utils->tensor_min_max(attention->key->getWeightDevice(), d_model * d_model, "device");

                // printf("Epoch: %d \n", i);
                // utils->tensor_min_max(prob, batch_size * seq_len * vocab_size, "device");

                // if (i == 2000)
                // {
                //     // utils->printFlatArray2D(modelParamaters.dweight_lm_head, vocab_size, d_model, "device");
                //     utils->printFlatArray2D(attention->embeddings->deviceEmbeddings, vocab_size, d_model, "device");
                // }

                modelParamaters.weight_lm_head = lm_head->getWeightDevice();

                modelParamaters.bias_lm_head = lm_head->getBaiasDevice();

                // std::cout << " LM head " << std::endl;
                // this->utils->printFlatArray3D(prob, batch_size, seq_len, vocab_size, false, "device");

                softmaxAcrossProballityCrossEntropyLoss(prob, batch.y);

                loss_per_batch += this->utils->averageFlatStrip2D(outCrossEntropyHost->get(), batch_size * seq_len);

                //  cout << "Loss per batch: " << this->utils->averageFlatStrip2D(outCrossEntropyHost, batch_size * seq_len) << endl;

                // if (debug)
                // {
                //     std::cout << " After sfotmax last two dimension " << std::endl;
                //     this->utils->printLastOneOf3D(prob, batch_size, seq_len, vocab_size);
                // }

                modelParamaters.token_id = batch.x;

                // int *a = attention->embeddings->getDeviceX();
                // int *b = modelParamaters.token_id;

                // std::cout << "getter: " << static_cast<void *>(a) << '\n';
                // std::cout << "token_id: " << static_cast<void *>(b) << '\n';

                // utils->printFlatArray2D(a, batch_size, seq_len, "device");
                // utils->printFlatArray2D(b, batch_size, seq_len, "device");
                // ----------- Lets gather overall paramaters here ---------- //
                modelParamaters.attention_head = attention->getParamaters();
                modelParamaters.lm_head = LinearParams{lm_head->getWeightDevice(), lm_head->getBaiasDevice()};
                modelParamaters.L = outCrossEntropyDevice->get();        // CE Out
                modelParamaters.y_actual = yHotEncodeDeviceOut->get();   // (B, T, vocab_size) from actual
                modelParamaters.y_predicted = DeviceSoftmaxBLout->get(); // (B, T, vocab_size) to predicted proballity

                modelParamaters.upsteram_out_softmax_ce = upstream_softmax_ce->get();

                modelParamaters.final_attn_tensor = attn_head_out->get(); // (B, T, d_model) on device
                modelParamaters.xt_device = ht_device->get();

                modelParamaters.dbias_lm_head_pred = dbias_lm_head_pred->get();

                // for dl_dw = delta h^T derived in flashback.md

                modelParamaters.dweight_lm_head = dl_dw_out_device->get();

                // these are just borrowed pointers
                modelParamaters.wt_lm_head_deice = w_out_d->get();

                modelParamaters.P_T_device_out = P_T_device_out->get();
                modelParamaters.V_T_device_out = V_T_device_out->get();

                // GDDR RAM
                modelParamaters.Uncontact_G_Upstream = Uncontact_G_Upstream->get();
                modelParamaters.upstream_grad_output_proj = Contact_G_Upstream->get();

                modelParamaters.dV = dV->get();
                modelParamaters.dP = dP->get();

                modelParamaters.ppt = ppt->get();

                modelParamaters.dQ = dQ->get();
                modelParamaters.dK = dK->get();

                modelParamaters.d_score_t = d_score_t->get();

                modelParamaters.WkT = WkT->get();
                modelParamaters.WvT = WvT->get();
                modelParamaters.WqT = WqT->get();

                modelParamaters.Wk = Wk->get();
                modelParamaters.WV = WV->get();
                modelParamaters.WQ = WQ->get();

                modelParamaters.qUp = qUp->get();
                modelParamaters.kUp = kUp->get();
                modelParamaters.vUp = vUp->get();

                modelParamaters.d_weight_q = d_weight_q->get();
                modelParamaters.d_weight_k = d_weight_k->get();
                modelParamaters.d_weight_v = d_weight_v->get();

                modelParamaters.d_bias_q = d_bias_q->get();
                modelParamaters.d_bias_k = d_bias_k->get();
                modelParamaters.d_bias_v = d_bias_v->get();

                modelParamaters.dqWt = dqWt->get();
                modelParamaters.dkWt = dkWt->get();
                modelParamaters.dvWt = dvWt->get();

                modelParamaters.G_x_hat = G_x_hat->get(); // net G_hat from the derivation

                modelParamaters.debeta = dbeta->get();
                modelParamaters.dgamma = dgamma->get();

                // Tranapose of weight of contact i.e output projection
                modelParamaters.WoT = WoT->get();
                modelParamaters.dl_dh_output = dl_dh_output->get();
                modelParamaters.d_add_residual_output = d_add_residual_output->get();

                modelParamaters.d_embedding = d_embedding->get();

                modelParamaters.d_weight_output_project = d_weight_output_project->get();
                modelParamaters.d_softmax_upstream = d_softmax_upstream->get();

                modelParamaters.dx = dx->get();

                // because the backprops needs to be done for each epoch.
                // we need to keep in mind that the things hurting performace like cuda malloc and everything declared
                // inside of the constructor of autograd engine is costly.
                // there is tradeoff between making things modular and fusing everything together.
                // Lets create a buffer for CPU/GPU memory in this class so that we dont overload the system and free it when the object is destroyed.

                autograd->backprop(modelParamaters);

                // if (debug)
                // {
                //     float *dh = (float *)malloc(d_model * d_model * sizeof(float));
                //     cudaMemcpy(dh, modelParamaters.d_weight_q, d_model * d_model * sizeof(float), cudaMemcpyDeviceToHost);
                //     cout << "Unupdated gradeint w" << endl;
                //     utils->printFlatArray2D(dh, d_model, d_model);

                //     free(dh);
                // }

                optimizer->invoke(modelParamaters, *adamMemConfig, epoch);

                // check if something is off with AadamW on sufrace

                // if (debug)
                // {
                //     float *dh = (float *)malloc(d_model * d_model * sizeof(float));
                //     cudaMemcpy(dh, modelParamaters.d_weight_q, d_model * d_model * sizeof(float), cudaMemcpyDeviceToHost);
                //     cout << "udpated gradeint w" << endl;
                //     utils->printFlatArray2D(dh, d_model, d_model);

                //     free(dh);
                // }

                debug = false;

                batch_count++;
            }
            if (i % 250 == 0)
            {

                float epoch_loss = batch_count > 0 ? loss_per_batch / batch_count : 0.0f;

                std::cout << "Epoch: " << i
                          << " Loss: " << epoch_loss << std::endl;
            }

            loss += loss_per_batch;
        }
    }
};
