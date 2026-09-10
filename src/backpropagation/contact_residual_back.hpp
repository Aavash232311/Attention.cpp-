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
     *
     * @important
     * @warning This method was added layer previously dl_dh was considered upstream gradient because I forgot this
     * contact projection. Now what I have done is make the Contact_G and gradient flowing deep in the attention head
     * as the gradient from this method of output project. Debugger flags green but remember debugger cannot debug
     * just everything just check the mathematical kernel opreations.
     *
     * For now I think this works but in future if we got the accuracy that is in the random threshold then we might need to check this out
     * adf035f06eb37c1312bea700bc41b334a62d2b4f
     */
    void outputProj()
    {
        wt_upstream(model_paramaters.attention_head.wo,
                    model_paramaters.WoT,
                    d_model,
                    d_model);

        if (debug)
        {
            cout << "dl_dh upstream before backpass of the attention head" << endl;
            float *G = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));

            cudaMemcpy(G, model_paramaters.dl_dh_output, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);

            utils->printFlatArray3D(G, batch_size, seq_len, d_model);

            free(G);
        }

        // This now releases what we call the G I am cooked but it is what it is hold tight.
        dl_dh_upstream(model_paramaters.dl_dh_output,
                       model_paramaters.WoT,
                       model_paramaters.Contact_G_Upstream,
                       batch_size,
                       seq_len,
                       d_model,
                       d_model);

        // for testing what I want to do is, copy that

        dbias(model_paramaters.dl_dh_output, // upstream gradient
              model_paramaters.attention_head.doutput_bias,
              batch_size,
              seq_len,
              d_model);


        if (debug)
        {
            pyDebuggerReleaseStage9();
        }
    }
};