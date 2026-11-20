//
// Created by avash on 11/20/26.
//

#include "interface.hpp"

AttentionInterface::AttentionInterface(const int d_model, const int num_heads, const int batch_size, const int seq_len, const bool drop_last, const int vocab_size, const std::vector<Batch> &data, const bool debug, const bool engine_run) :
d_model(d_model), num_heads(num_heads), vocab_size(vocab_size), batch_size(batch_size), seq_len(seq_len), drop_last(drop_last), data(data), debug(debug), engine_run(engine_run) {
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


        DeviceSoftmaxBLin = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, vocab_size}, DEVICE);
        // Tensor<float> DeviceSoftmaxBLout(vector<int>{batch_size, seq_len, vocab_size}, DEVICE);
        DeviceSoftmaxBLout = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, vocab_size}, DEVICE);

        outCrossEntropyDevice = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len}, DEVICE); // (N) across all of the BT we will have the loss.
        cudaMemset(outCrossEntropyDevice->get(), 0, batch_size * seq_len * sizeof(float));        // make it zero for the things that we are not touching.

        // YOU DO NEED THIS PART BECAUSE ANOTHER KERNEL FOR THE CROSS ENTROPY LOSS WILL BE USING THIS.
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

        Un_contact_G_Upstream = std::make_unique<Tensor<float>>(std::vector<int>{batch_size, seq_len, num_heads, head_dim}, DEVICE);

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

        delta_beta = std::make_unique<Tensor<float>>(std::vector<int>{d_model}, DEVICE);
        delta_gamma = std::make_unique<Tensor<float>>(std::vector<int>{d_model}, DEVICE);

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

void AttentionInterface::softmaxAcrossProbabilityCrossEntropyLoss(float *probability, int *y) const {

    // this accounts for whatever we are doing softmax on being greater than 32 i.e warp size.
    // THIS IS SOFTMAX ACROSS the logits (B, T, vocab_size)

    /*
        DeviceSoftmaxBLout: Probability from logits (predicted)
        yHotEncodeDeviceOut: Actual Y (B, T) -> (B, T, vocab_size)
        outCrossEntropyDevice: CE out flat list
    */

    if (debug)
        attention->releaseBin(probability, "lm_head_out.bin", batch_size * seq_len * vocab_size);

    softmax2D(
        probability,
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

}


void AttentionInterface::forward(std::string input, std::vector<char> &charPool, const int gen) {
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
            reinterpret_cast<void **>(&softmaxedProb),
            prompt_seq_len * vocab_size * sizeof(float));

        auto *softmaxedProbHost = static_cast<float *>(malloc(
            prompt_seq_len * vocab_size * sizeof(float)));

        for (int i = 0; i < gen; ++i)
        {
            const vector<int> window(
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

            const int offset = (prompt_seq_len - 1) * vocab_size;

            int index = 0;
            float maxProb = softmaxedProbHost[offset];

            for (int j = 1; j < vocab_size; ++j)
            {
                if (const float currentProb = softmaxedProbHost[offset + j]; currentProb > maxProb)
                {
                    maxProb = currentProb;
                    index = j;
                }
            }
            encodedInput.push_back(index);

            cudaFree(tokens);
        }

        const auto decode = helper->decoder(encodedInput);

        for (const char i : decode)
        {
            cout << i;
        }

        cout << "\n";

        free(softmaxedProbHost);
        cudaFree(softmaxedProb);
}

void AttentionInterface::train(const int epoch) {
            float loss = 0.0f;

        for (int i = 0; i < epoch; ++i)
        {
            float loss_per_batch = 0.0f;
            int batch_count = 0;

            for (const auto batch : data)
            {
                // Shape (B, T, C) only pointer dependent upon the channel dimension is here
                // And there is an illegal memory access somewhere here.

                // current batch maybe in GPU/CPU

                float *x = attention->forward(batch);

                cudaMemcpy(attn_head_out->get(), x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);

                float *prob = lm_head->forward(x); // Shape (B, T, vocab_size) x is not changed here.
                modelParameters.weight_lm_head = lm_head->getWeightDevice();

                modelParameters.bias_lm_head = lm_head->getBiasDevice();
                softmaxAcrossProbabilityCrossEntropyLoss(prob, batch.y);

                loss_per_batch += this->utils->averageFlatStrip2D(outCrossEntropyHost->get(), batch_size * seq_len);

                modelParameters.token_id = batch.x;
                modelParameters.attention_head = attention->getParamaters();
                modelParameters.lm_head = LinearParams{.Weight = lm_head->getWeightDevice(), .Bias = lm_head->getBiasDevice()};
                modelParameters.L = outCrossEntropyDevice->get();        // CE Out
                modelParameters.y_actual = yHotEncodeDeviceOut->get();   // (B, T, vocab_size) from actual
                modelParameters.y_predicted = DeviceSoftmaxBLout->get(); // (B, T, vocab_size) to predicted probability

                modelParameters.upsteram_out_softmax_ce = upstream_softmax_ce->get();

                modelParameters.final_attn_tensor = attn_head_out->get(); // (B, T, d_model) on device
                modelParameters.xt_device = ht_device->get();

                modelParameters.dbias_lm_head_pred = dbias_lm_head_pred->get();

                // for dl_dw = delta h^T derived in flashback.md

                modelParameters.dweight_lm_head = dl_dw_out_device->get();

                // these are just borrowed pointers
                modelParameters.wt_lm_head_deice = w_out_d->get();

                modelParameters.P_T_device_out = P_T_device_out->get();
                modelParameters.V_T_device_out = V_T_device_out->get();

                // G-DDR RAM
                modelParameters.Uncontact_G_Upstream = Un_contact_G_Upstream->get();
                modelParameters.upstream_grad_output_proj = Contact_G_Upstream->get();

                modelParameters.dV = dV->get();
                modelParameters.dP = dP->get();

                modelParameters.ppt = ppt->get();

                modelParameters.dQ = dQ->get();
                modelParameters.dK = dK->get();

                modelParameters.d_score_t = d_score_t->get();

                modelParameters.WkT = WkT->get();
                modelParameters.WvT = WvT->get();
                modelParameters.WqT = WqT->get();

                modelParameters.Wk = Wk->get();
                modelParameters.WV = WV->get();
                modelParameters.WQ = WQ->get();

                modelParameters.qUp = qUp->get();
                modelParameters.kUp = kUp->get();
                modelParameters.vUp = vUp->get();

                modelParameters.d_weight_q = d_weight_q->get();
                modelParameters.d_weight_k = d_weight_k->get();
                modelParameters.d_weight_v = d_weight_v->get();

                modelParameters.d_bias_q = d_bias_q->get();
                modelParameters.d_bias_k = d_bias_k->get();
                modelParameters.d_bias_v = d_bias_v->get();

                modelParameters.dqWt = dqWt->get();
                modelParameters.dkWt = dkWt->get();
                modelParameters.dvWt = dvWt->get();

                modelParameters.G_x_hat = G_x_hat->get(); // net G_hat from the derivation

                modelParameters.debeta = delta_beta->get();
                modelParameters.dgamma = delta_gamma->get();

                // Transpose of weight of contact i.e output projection
                modelParameters.WoT = WoT->get();
                modelParameters.dl_dh_output = dl_dh_output->get();
                modelParameters.d_add_residual_output = d_add_residual_output->get();

                modelParameters.d_embedding = d_embedding->get();

                modelParameters.d_weight_output_project = d_weight_output_project->get();
                modelParameters.d_softmax_upstream = d_softmax_upstream->get();

                modelParameters.dx = dx->get();

                // because the backdrops needs to be done for each epoch.
                // we need to keep in mind that the things hurting performance like cuda malloc and everything declared
                // inside of the constructor of autograd engine is costly.
                // there is tradeoff between making things modular and fusing everything together.
                // Lets create a buffer for CPU/GPU memory in this class so that we dont overload the system and free it when the object is destroyed.

                autograd->backprop(modelParameters);
                optimizer->invoke(modelParameters, *adamMemConfig, epoch);

                // check if something is off with AadamW on surface

                debug = false;

                batch_count++;
            }
            if (i % 250 == 0)
            {

                const float epoch_loss = batch_count > 0 ? loss_per_batch / static_cast<float>(batch_count) : 0.0f;

                std::cout << "Epoch: " << i
                          << " Loss: " << epoch_loss << std::endl;
            }

            loss += loss_per_batch;
        }
}