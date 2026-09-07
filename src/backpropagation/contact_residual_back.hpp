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

extern "C" void wt_upstream(float *w, float *wt, int c1, int c2);
extern "C" void dl_dh_upstream(float *detla, float *wt, float *out, int B, int T, int C, int vocab_size);
extern "C" void dbias(float *G, float *dbias, int B, int T, int C);

// Contact and add residual backpropagation

class ContactResidualBack : virtual public AutoGradEngine
{
public:
    ContactResidualBack(int d_model, int vocab_size, int num_heads,
                        int seq_len, int batch_size, bool debug)
        : AutoGradEngine(d_model, vocab_size, num_heads, seq_len, batch_size, debug)
    {
    }


    /**
     * @class ContactResidualBack
     * @brief Backpropagation through output projection i.e contact paramaters
     *
     * @param Upstream Gradient pointer to be modified 
     *
     * @note Contact function is a linear function with weight and biases since we are using our custom linear class.
     *      Let us consider a linear function output_porjeciton = attention Wo + b
     *      Where Wo is the contact paramater and b is the bias.
     * 
     *     Wq, Wk, and Wv have (C, C) shape we can re-use that
     *     If these are gradients then no, AdamW or Adam later will need these values
     * 
     * @warning This should always be called right after we get gradients from FFN or LM head. Right now we are igonoring FNN so its LM head.
     */
    void outputProj()
    {
        wt_upstream(model_paramaters.attention_head.wo, model_paramaters.attention_head.woT, d_model, d_model);
        dl_dw_upstream(model_paramaters.Contact_G_Upstream, model_paramaters.attention_head.woT, model_paramaters.attention_head.dattn_out, batch_size, seq_len, d_model, d_model);
        dbias(model_paramaters.attention_head.wo_bias, model_paramaters.attention_head.doutput_bias, batch_size, seq_len, d_model);

        if (debug)
        {
            pyDebuggerReleaseStage9();
        }
    }
};