#pragma once
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./linear_layer_local_and_up.hpp"

#include "../include/utils.hpp"
#include "../include/linear.hpp"
#include "../include/p_head.hpp"
#include "../include/cache_in.hpp"
#include "../include/cache_out.hpp"
#include "../include/netattention.hpp"
#include "../include/attention_params.hpp"
#include "../include/single_embeddings.hpp"

// ----------- Backpropgation ------------------------
extern "C" void upstream_dl_dz(float *actual, float *predicted, float *delta, int B, int T, int C);
extern "C" void lm_head_transpose_h(float *h, float *out, int B, int T, int C);
extern "C" void dl_dw_upstream(float *h_t, float *delta, float *out, int B, int T, int C, int vocab_size);
extern "C" void wt_upstream(float *w, float *wt, int d_model, int vocab_size);
extern "C" void dl_dh_upstream(float *detla, float *wt, float *out, int B, int T, int C, int vocab_size, bool sync);
extern "C" void dbias(float *G, float *dbias, int B, int T, int C, bool sync);
// ---- Paramaters for our custom backgrad engine -----

/*
    For performace reason we do not move the data between VRAM and RAM.
    So to reduce the cudaMemcpy we use the globally allocated memory
    To print and debug this in low level code is is equally important
    we create a temp array in CPU to see it.

*/
template <typename PrintFunc>
void DebugBuffer(size_t count, PrintFunc print)
{
    float *temp = new float[count];

    // Fill temp somehow (e.g. cudaMemcpy)

    print(temp, count);

    delete[] temp;
}

// The chain rule
class AutoGradEngine
{
    // welcome to my calculas class

protected:
    FlashAttentionPointers model_paramaters;

    bool debug = true;

    // ------- For utility purpose -------------
    int d_model;
    int vocab_size;
    int num_heads;
    int seq_len;
    int batch_size;
    int head_dim;

    // ---------- Handy methods -----------
    std::unique_ptr<Utility> utils = std::make_unique<Utility>();
        std::unique_ptr<LinearLayerBackpropagation> linearBack;

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

    // REMEMBER BESIDE ME NO ONE WILL EVEERRRRR READ THIS CODE
    // IF ITS DIRTY THEN I WILL HANDLE ITTTT.

    // This is from the endless marching of time
    // from the universe.
    // sure dbias is delta but this won't sum it don't know
    // why did I wrote it like that.
    void dl_dz_upstream_gradient( // delta
        float *actual,            // (B, T, vocab_size) on device
        float *predicted,         // (B, T, vocab_size) on device
        float *delta,             // (B, T, vocab_size) on device upstream gradient
        float *delta_host,
        int B,
        int T,
        int vocab_size)
    {
        // interfaceback.md derivation using the chain rule of derivative
        upstream_dl_dz( // dl_dz is partial derivative
            actual,
            predicted,
            delta,
            B,
            T,
            vocab_size);

        // upstream gradient to host, we can keep this in the device but we will fix this later, first goal is to get the result right
        cudaMemcpy(delta_host, delta, B * T * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        // delta_host = partial L / partial z
    }

public:
    AutoGradEngine(
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

        this->head_dim = d_model / num_heads;

        linearBack = std::make_unique<LinearLayerBackpropagation>(
            batch_size,
            seq_len,
            d_model,
            vocab_size,
            debug);

        // NOTE- Memory allocation in RAM or VRAM is done per epoch if done here
        // huritng the performace, allocate and re-use ones from the attention
        // consturcotr and pass as a buffer.
    }

    // ---------- child methods -----------------

    virtual void opv_upstream_gradient(Tensor4 shape) {}

    virtual void pyDebuggerReleaseStage1() {}
    virtual void pyDebuggerReleaseStage2() {}
    virtual void pyDebuggerReleaseStage3() {}
    virtual void pyDebuggerReleaseStage4() {}
    virtual void pyDebuggerReleaseStage5() {}
    virtual void pyDebuggerReleaseStage6() {}
    virtual void pyDebuggerReleaseStage7() {}
    virtual void pyDebuggerReleaseStage8() {}
    virtual void pyDebuggerReleaseStage9() {}
    virtual void pyDebuggerReleaseStage10() {}
    virtual void pyDebuggerReleaseStage11() {}

    // Backpropagation along Linear layer, Normalization
    virtual void NormLinearNet() {}
    virtual void weightTransposeAttn() {}
    virtual void outputProj() {}

    void backprop(
        const FlashAttentionPointers &paramaters)
    {
        this->model_paramaters = paramaters;

        // if (debug)
        // {
        //     utils->printFlatArray2D(
        //         model_paramaters.w_host,
        //         d_model,
        //         vocab_size);
        // }

        dl_dz_upstream_gradient(
            paramaters.y_actual, // Note:- these are on device
            paramaters.y_predicted,
            paramaters.dl_dz_out_device, // delta
            paramaters.dl_dz_out_host,   // writes DELTA HERE
            batch_size,
            seq_len,
            vocab_size);

        // if (debug) {
        //     // before transpose the shape if (B, T, C)
        //     utils->printFlatArray3D(paramaters.h, batch_size, seq_len, d_model);
        // }

        if (debug)
            pyDebuggerReleaseStage1();

        if (debug)
        {

            // float *h = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));
            // cudaMemcpy(h, model_paramaters.device_h, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

            // std::cout << "x from the host " << std::endl;
            // utils->printFlatArray3D(model_paramaters.h, batch_size, seq_len, d_model);

            // std::cout << "x from the device" << std::endl;
            // utils->printFlatArray3D(h, batch_size, seq_len, d_model);

            // free(h);
        }

        linearBack->backward(
            paramaters.x,
            model_paramaters.dl_dz_out_device,
            paramaters.weight_lm_head,
            paramaters.wt_lm_head_deice,
            paramaters.xt_device,
            paramaters.dweight_lm_head,
            paramaters.dbias_lm_head_pred,
            paramaters.dl_dh_output,
            true);

        if (debug)
            pyDebuggerReleaseStage3();

        // Now we will take care about the output_proj
        // Contact_G_Upstream = upstream gradient from the interface

        outputProj();

        // Calls the backpropagation for the attention head items
        opv_upstream_gradient({batch_size, seq_len, vocab_size});
        // Ignoring the FFN for now we will call the flash attention layer.

        // if (debug)
        // {

        //     // here we swap the dimension from (B, T, d_model) to (B, d_model, T)
        //     std::cout << "h^T Shape (B, C, T)" << std::endl;
        //     utils->printFlatArray3D(paramaters.h, batch_size, d_model, seq_len);
        // }

        if (debug)
        {
            // std::cout << "Predicted" << std::endl;
            // DebugBTCFlatArray3D(paramaters.y_predicted, batch_size, seq_len, vocab_size);

            // std::cout << "Actual" << std::endl;
            // DebugBTCFlatArray3D(paramaters.y_actual, batch_size, seq_len, vocab_size);

            // std::cout << "dl_dz detla" << std::endl;
            // utils->printLastOneOf3D(paramaters.dl_dz_out_host,
            //     batch_size,
            //     seq_len,
            //     vocab_size
            // );

            // std::cout << "Actual proballity" << std::endl;
            // utils->printFlatArray3D(paramaters.y_actual, batch_size, seq_len, vocab_size);

            // std::cout << "Predicted proballity" << std::endl;
            // utils->printFlatArray3D(paramaters.y_predicted, batch_size, seq_len, vocab_size);
        }

        debug = false;
    }
};