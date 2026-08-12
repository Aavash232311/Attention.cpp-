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
    P = softmax(x) shape (batch_size * num_heads * seq_len * seq_len )
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
    // G = gradient from the interface (that's what I like to call even though it may not be std)
    // O = PV is there.
    // p = softmax(x)

    // dV = P^T G
    // dP = GV^T

    void opv_upstream_gradient(
        Tensor4 shape) override
    {
        if (debug == true)
        {
            std::cout << "After softmax from autograd eignine " << sizeof(P) << std::endl;
            utils->print2DMatrixLastTwo(model_paramaters.attention_head.P, batch_size, num_heads, seq_len, seq_len);
        }
    }
};