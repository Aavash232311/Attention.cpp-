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
extern "C" void dl_dh_upstream(float *detla, float *wt, float *out, int B, int T, int C, int vocab_size);
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

    // h^T
    void gradient_linear(
        float *h_host, // input (B, T, d_model) and after the lm head (B, T, vocab_size)
        float *h_device,
        float *h_out, // device (B, C, T) shape for delta h^T
        float *delta, // (B, T, vocab_size)
        int B,
        int T,
        int d_model,
        int vocab_size)
    {
        // Copy from host to device
        cudaMemcpy(h_device, h_host, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);

        // h^T
        lm_head_transpose_h(
            h_device,
            h_out,
            batch_size,
            seq_len,
            d_model);

        // copy back to host, just to check and debug, else we do not need to copy back and fourh between
        // different memories that is costly, burnout is expected at this complexity and lines of code.
        cudaMemcpy(h_host, h_out, batch_size * d_model * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void dl_dw_upstream_gradient(
        float *delta_device, // (B, T, vocab_size)
        float *h_device,     // (B, C, T)   Here: h = h^T (transposed by the derivation)
        float *out_device,   // (B, C, vocab_size)
        float *out_host,     //  B, C, vocab_size)
        int B,
        int T,
        int C,
        int vocab_size)
    {

        dl_dw_upstream(
            h_device,
            delta_device,
            out_device,
            B,
            T,
            C,
            vocab_size);

        // write to host, we have a debugger release from which we can copy to just just for the sake of releasing
        cudaMemcpy(out_host, out_device, batch_size * vocab_size * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // this is something that you have written in Java already sometime many years ago
    // just back propagation in output layer why because this is linear

    // before lm_head -> (B, T, C) output is  (B, T, vocab_size)
    // I may not be so smart, atleast now I understand the defination

    void wt_upstream_gradient(
        float *w_host,
        float *w_device, // (d_model, vocab_size)
        float *w_out_d,  // (vocab_size, d_model)
        int d_model,
        int vocab_size)
    {
        // copy form host w
        cudaMemcpy(w_device, w_host, d_model * vocab_size * sizeof(float), cudaMemcpyHostToDevice);

        wt_upstream(
            w_device,
            w_out_d,
            d_model,
            vocab_size);
    }

    void dl_dh_upstream_gradient(
        float *delta,        //  (vocab_size, d_model)
        float *wt,     //  (B, T, vocab_size)
        float *out_dl_dh, // the real upstream gradient G, I accidently thought its dl_dw
        int batch_size,
        int seq_len,
        int d_model,
        int vocab_size)
    {
        dl_dh_upstream(
            delta,
            wt,
            out_dl_dh,
            batch_size,
            seq_len,
            d_model,
            vocab_size);
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


    // Backpropagation along Linear layer, Normalization 
    virtual void NormLinearNet() {}
    virtual void weightTransposeAttn() {}

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

        gradient_linear(
            paramaters.h, //
            paramaters.device_h,
            paramaters.device_out_h,     // out h^T
            paramaters.dl_dz_out_device, // delta on device
            batch_size,
            seq_len,
            d_model,
            vocab_size);

        // delta h^T for weights

        /*
            void dl_dw_upstream_gradient(
        float *delta_device, // (B, T, vocab_size)
        float *h_device,     // (B, C, T)   Here: h = h^T (transposed by the derivation)
        float *out_device,   // (B, C, vocab_size)
        float *out_host,     //  B, C, vocab_size)
        int B,
        int T,
        int C,
        int vocab_size)
    {

        */
        // I am sorry for the confusing name
        // remember no one EVERRR is reading this
        // so on my lead only for me.
        dl_dw_upstream_gradient(
            paramaters.dl_dz_out_device, // delta device
            paramaters.device_out_h,     // its going to be h^T after transpose kernel writes to this kernel
            paramaters.dl_dw_device,     // for out
            paramaters.dl_dw_host,
            batch_size,
            seq_len,
            d_model,
            vocab_size);

        if (debug)
            pyDebuggerReleaseStage2();

        wt_upstream_gradient(
            paramaters.w_host,
            paramaters.w_device,
            paramaters.wt_out_d,
            d_model,
            vocab_size);

        dl_dh_upstream_gradient(
            paramaters.dl_dz_out_device,         // delta
            paramaters.wt_out_d,                 // w^t
            model_paramaters.Contact_G_Upstream, // (B, T, C)
            batch_size,
            seq_len,
            d_model,
            vocab_size);

        if (debug)
            pyDebuggerReleaseStage3();

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