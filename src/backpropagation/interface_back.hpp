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

#pragma once


// ----------- Backpropgation ------------------------
extern "C" void upstream_dl_dz(float *actual, float *predicted, float *delta, int B, int T, int C);
extern "C" void lm_head_transpose_h(float *h, float *out, int B, int T, int C);
extern "C" void dl_dw_upstream(float *h_t, float *delta, float *out, int B, int T, int C, int vocab_size);
extern "C" void wt_upstream(float *w, float *wt, int d_model, int vocab_size);
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

    NetAttentionParamaters model_paramaters;

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
        float *w_out_d,
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

    /**
     * @class pyDebuggerReleaseStage1
     * @brief Releases paramaters y, y predicted, delta, CE+softmax backpropagation, h (from lm head)
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Call this only after dl_dz_upstream_gradient() and before gradient_linear <- tranposes h^T
     * @warning Do not call this before the above methods are called as they tend to write garbage data.
     */
    void pyDebuggerReleaseStage1()
    {
        // release those paramaters so that we can cross verify the kernel in python
        // now I wnat to debug along rather than estimating by judging few rows and cols.
        // pointers are manupluated and copied to the host.

        // y_actual and y_predicted are in GDDR VRAM
        // for debugging its fine to allocate here and free it

        float *host_y_actual = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));
        float *host_y_prediced = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));

        // happens only one that that also for debugging so it should be fine.
        cudaMemcpy(host_y_actual, model_paramaters.y_actual, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_y_prediced, model_paramaters.y_predicted, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        bulkRelease<float>({{host_y_actual, batch_size * seq_len * vocab_size, "y_actual.bin"},
                            {host_y_prediced, batch_size * seq_len * vocab_size, "y_prediced.bin"},
                            {model_paramaters.dl_dz_out_host, batch_size * seq_len * vocab_size, "delta.bin"},
                            {model_paramaters.h, batch_size * seq_len * d_model, "h.bin"}});

        free(host_y_actual);
        free(host_y_prediced);
    }

    /**
     * @class pyDebuggerReleaseStage2
     * @brief Releases paramaters h^T and dl/dw = del h^T
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Call this only after gradient_linear and dl_dw_upstream_gradient
     * @warning Do not call this before the above methods are called as they tend to write garbage data.
     */
    void pyDebuggerReleaseStage2()
    {
        // After the gradient_linear() gets called model_paramaters.h gets written
        bulkRelease<float>({{model_paramaters.h, batch_size * seq_len * d_model, "h_t.bin"},
                            {model_paramaters.dl_dw_host, batch_size * d_model * vocab_size, "dl_dw.bin"}}); // out delta h^T binary
                                                                                                             // second stage release for the autograd engine.
    }

    /**
     * @class pyDebuggerReleaseStage3
     * @brief Releases paramaters w^t
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Can call this in any order because w is independent of the result from pervious things like h.
     */

    void pyDebuggerReleaseStage3()
    {
        // copy w^T to host for the python script to read the binary

        float *wt_host = (float *)malloc(d_model * vocab_size * sizeof(float));

        cudaMemcpy(wt_host, model_paramaters.wt_out_d, d_model * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);


        bulkRelease<float>(
            {{wt_host, d_model * vocab_size, "wt.bin"},
             {model_paramaters.w_host, d_model * vocab_size, "w.bin"}});

        free(wt_host);
    }


    /*
    --------------------- delta w^t = upstream gardient for the attention head -------------------------------------
                                backpropagation through attention head
    */

    // Note:- out = x + attn(x) is just adding two pices together we will first do the backpropagation in the attention mechanism
    // Lets go for S P O backpropagation here, its a mess but it is what it is.


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

    void backprop(
        const NetAttentionParamaters &paramaters)
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

        if (debug)
            pyDebuggerReleaseStage3();

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