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
extern "C" void TransposeKey(float *arr, float *out, int num_heads, int head_dim, int batch_size, int seq_len, bool reverse);

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
        int x,
        int y,
        int z,
        int z1)
    {
        cudaMemcpy(arr_d, arr_h, x * y * z * z1 * sizeof(float), cudaMemcpyHostToDevice);

        TransposeKey(
            arr_d, // it will replace that same variable variable
            arr_h,
            x,
            y,
            z,
            z1,
            false);

        cudaMemcpy(arr_h, arr_d, x * y * z * z1 * sizeof(float), cudaMemcpyDeviceToHost);
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
        // Transposing P^T and V^T because they share common kernel logic.
        float *PT = model_paramaters.P_T_device;

        if (debug) // release the default pointer.
            pyDebuggerReleaseStage4();

        // For P^T
        transpose4DLastTwo(
            model_paramaters.attention_head.P, // (batch_size * num_heads * seq_len * seq_len )
            model_paramaters.P_T_device,
            batch_size, // according to the shape of P
            num_heads,
            seq_len,
            seq_len);

        // For V^T
        transpose4DLastTwo(
            model_paramaters.attention_head.V, // (B, n_head, T, head_dim)
            model_paramaters.V_T_device,
            batch_size, // according to the shape of P
            num_heads,
            seq_len,
            seq_len);

        if (debug)
            pyDebuggerReleaseStage5();

        // if (debug == true)
        // {
        //     std::cout << "After softmax " << sizeof(model_paramaters.attention_head.P) << std::endl;
        //     utils->print2DMatrixLastTwo(model_paramaters.attention_head.P, batch_size, num_heads, seq_len, seq_len);
        // }

        // now model_paramaters.attention_head.P this pointer is transposed here.
    }
};