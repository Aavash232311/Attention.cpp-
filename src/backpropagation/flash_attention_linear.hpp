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
extern "C" void matmul3d2d(float *A, float *B, float *C, int a, int b, int c, int d);
// G_kx0 total upstream gradient and Linear Layer, add-residual back propagation here.
class FlashAttentionLinear : virtual public AutoGradEngine
{

private:
    void copyWeightQKVtoDevice()
    {
        // These data are in host automatically from the Linear class
        // we have the buffer from interface class now we need to copy that

        // from attention pointer has the thing inside of CPU
        // copy them all to GPU

        // K and V are allocated elsewhere just re-using this pointer

        cudaMemcpy(model_paramaters.Wk, model_paramaters.attention_head.host_WK, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(model_paramaters.WQ, model_paramaters.attention_head.host_WQ, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(model_paramaters.WV, model_paramaters.attention_head.host_WV, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);

        wt_upstream(model_paramaters.Wk, model_paramaters.WkT, d_model, d_model);
        wt_upstream(model_paramaters.WQ, model_paramaters.WqT, d_model, d_model);
        wt_upstream(model_paramaters.WV, model_paramaters.WvT, d_model, d_model);

        if (debug)
            pyDebuggerReleaseStage8();
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