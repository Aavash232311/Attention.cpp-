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

/*
--------------------- delta w^t = upstream gardient for the attention head -------------------------------------
                            backpropagation through attention head
*/

// Note:- out = x + attn(x) is just adding two pices together we will first do the backpropagation in the attention mechanism
// Lets go for S P O backpropagation here, its a mess but it is what it is.
// Note:- FFN is ignored for the time being.

/*
    Shapes: dl_dw (B, C, vocab_size)

*/

class FlashAttention : public AutoGradEngine
{
private:
public:
    FlashAttention(int d_model, int vocab_size, int num_heads,
                   int seq_len, int batch_size, bool debug)
        : AutoGradEngine(d_model, vocab_size, num_heads, seq_len, batch_size, debug)
    {
    }

    // dV = P^T G
    // dP = GV^T

    void opv_upstream_gradient(
        float *G_device,
        float *G_host,
        Tensor3 shape) override
    {
        if (debug)
        {
            float* PV = model_paramaters.attention_head.O;
            utils->print2DMatrixLastTwo(PV, batch_size, num_heads, seq_len, head_dim);
            
        }
    }
};