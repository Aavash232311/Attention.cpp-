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

// Re-use this from linear.hpp
extern "C" void TransposeKey(int num_heads, int head_dim, float *arr, float *out, int M, int N, int K, bool reverse);

class FlashAttention : public AutoGradEngine
{
public:
    /**
     * @class FlashAttention: AutoGradEngine
     * @brief Transposes last two dimension of 4D tensor. 
     *
     * ransposes last two dimension of 4D tensor. 
     *
     * @note Re-used Kernel logic from linear.hpp
     */

    void transpose4DLastTwo(
        float *arr_h,
        float *arr_d,
        int N)
    {
        cudaMemcpy(arr_d, arr_h, N * sizeof(float), cudaMemcpyHostToDevice);

        TransposeKey(
            num_heads,
            head_dim,
            arr_d, // it will replace that same variable variable
            arr_h,
            batch_size,
            head_dim,
            seq_len,
            false);

        cudaMemcpy(arr_h, arr_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

private: // Note-: very limied kernel opreations here so for readability I am passing args and params.
    void dl_dv_gradient(

    )
    {
    }

    void dl_dp_gradient()
    {
    }

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
        // if (debug == true)
        // {
        //     utils->print2DMatrixLastTwo(model_paramaters.attention_head.P, batch_size, num_heads, seq_len, seq_len);
        // }
    }
};