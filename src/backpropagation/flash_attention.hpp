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
// Note:- Python sanity check has helped me spot different kernel bugs

/*
    Shapes: dl_dw (B, C, vocab_size)
    P = softmax(x) shape (batch_size * num_heads * seq_len * seq_len )
*/

// Re-use this from linear.hpp
extern "C" void TransposeKey(float *arr, float *out, int num_heads, int head_dim, int batch_size, int seq_len, bool reverse);
// from utils.cu
extern "C" void multiHeadedAttention(float *ws, float *out, int num_head, int head_dimension, int batch_size, int d_model, int seq_len);
extern "C" void MatMul4D(float *A, float *B, float *C, int a, int b, int c, int d, int e);
extern "C" void softmaxBackGradKernel(float *P, float *dY, float *out, int N, int batch_size, int seq_len, int n_head);

class FlashAttention : public AutoGradEngine
{
public:
    float *d_scores_device;
    /**
     * @class FlashAttention: AutoGradEngine
     * @brief Transposes last two dimension of 4D tensor PT
     * @note You are unwrapping the tensor in the kernel in a different sense so this x, y, z, z1 generic passing wont work here.
     * ransposes last two dimension of 4D tensor.
     *
     * @note Re-used Kernel logic from linear.hpp
     */

    void transpose4DLastTwoPT(
        float *arr_h,
        float *arr_d,
        float *arr_d_out,
        int x,
        int y,
        int z,
        int z1)
    {
        cudaMemcpy(arr_d, arr_h, x * y * z * z1 * sizeof(float), cudaMemcpyHostToDevice);

        TransposeKey(
            arr_d,
            arr_d_out,
            y,  // num_heads
            z1, // head_dim  (last dim, the one being moved)
            x,  // batch_size
            z,  // seq_len
            false);

        cudaMemcpy(arr_h, arr_d_out, x * y * z * z1 * sizeof(float), cudaMemcpyDeviceToHost);
    }

    /**
     * @class FlashAttention: AutoGradEngine
     * @brief Transposes last two dimension of 4D tensor VT
     *
     * ransposes last two dimension of 4D tensor.
     * @note You are unwrapping the tensor in the kernel in a different sense so this x, y, z, z1 generic passing wont work here.
     * @note Re-used Kernel logic from linear.hpp
     */

    void transpose4DLastTwoVT(
        float *arr_h,
        float *arr_d,
        float *arr_d_out,
        int x,
        int y,
        int z,
        int z1)
    {
        cudaMemcpy(arr_d, arr_h, x * y * z * z1 * sizeof(float), cudaMemcpyHostToDevice);

        TransposeKey(
            arr_d,
            arr_d_out,
            x,
            y,
            z,
            z1,
            false);

        cudaMemcpy(arr_h, arr_d_out, x * y * z * z1 * sizeof(float), cudaMemcpyDeviceToHost);
    }

private: // Note-: very limied kernel opreations here so for readability I am passing args and params.
    /*
        Note:- Upstream gradient G, ignoring the FFN for now, even if not ignored this step would be still the same
        if I am not wrong ofcourse because FFN are linears.

        Shape of G : (B, T, C)
        Shaoe of PT: (batch_size * num_heads * seq_len * seq_len )

        Uncontact G for matrix multiplication. Remember uncontact does not changes the value,
        just re-arranges the tensor for us to be able to do the matrix multiplication easily.

    */

    void multiHeadG(
        float *G, // (B, T, C)
        // note: its been months I don't remember how I wrote forward pass :)
        float *G_multi_headed, // (B, n_head, seq_len, head_dim)
        int num_head,
        int head_dim,
        int batch_size,
        int d_model,
        int seq_len)
    {
        multiHeadedAttention(
            G,
            G_multi_headed,
            num_head,
            head_dim,
            batch_size,
            d_model,
            seq_len);
    }

    void dv(
        float *PT, // (batch_size, num_heads, seq_len, seq_len )
        float *G,  // (B, n_head, seq_len, head_dim)
        float *dv, // (batch_size, num_heads, seq_len, head_dim)
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim)
    {
        MatMul4D(
            PT,
            G,
            dv,
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            head_dim);
    }

    void dp(
        float *VT, // (B, n_head, head_dim, T)
        float *G,  // (B, n_head, T, head_dim)
        float *dp, //  (B, n_head, T, T)
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim)
    {
        MatMul4D( // rememeber the order.
            G,
            VT,
            dp,
            batch_size,
            num_heads,
            seq_len,  // c
            head_dim, // d
            seq_len); // e
    }

    // Now we will have to deal with softmax part.
    // Here G is dl_dh if I am not wrong again I am old
    void softmaxBackGrad(
        float *P,  // (batch_sieq, num_head, seq_len, seq_len)
        float *dY, // shape (batch_size, seq_len, num_head, head_dim)
        float *out,
        int N, // I belive this is supposed to be N of P
        int btach_size,
        int seq_len,
        int n_head)
    {
        softmaxBackGradKernel(
            P,
            dY,
            out,
            N,
            batch_size,
            seq_len,
            n_head);
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

    // dV = P^T G, order must match here.
    // dP = GV^T

    void opv_upstream_gradient(
        Tensor4 shape) override
    {
        // Transposing P^T and V^T because they share common kernel logic.

        if (debug) // release the default pointer.
            pyDebuggerReleaseStage4();

        // For P^T
        transpose4DLastTwoPT(
            model_paramaters.attention_head.P, // (batch_size * num_heads * seq_len * seq_len )
            model_paramaters.P_T_device,
            model_paramaters.P_T_device_out,
            batch_size, // according to the shape of P
            num_heads,
            seq_len,
            seq_len);

        // For V^T
        transpose4DLastTwoVT(
            model_paramaters.attention_head.V, // (B, n_head, T, head_dim)
            model_paramaters.V_T_device,
            model_paramaters.V_T_device_out, // (B, n_head, head_dim, T)
            num_heads,                       // according to the shape of P
            head_dim,
            batch_size,
            seq_len);

        // the upstream gradient G from our LM head ignoring the FFN is of shape (B, T, C)
        // we need to re-arrange in terms of (batch_size, seq_len, num_head, head_dim)
        // Remember:- this does not changes the values just the way of writing it, its flat anyway.
        multiHeadG(
            model_paramaters.Contact_G_Upstream,   // (B, T, C)
            model_paramaters.Uncontact_G_Upstream, // (B, n_head, seq_len, head_dim)
            num_heads,
            head_dim,
            batch_size,
            d_model,
            seq_len);

        // now we are doing to multiuply P^T G and G V^T

        dv(
            model_paramaters.P_T_device_out,       // PT
            model_paramaters.Uncontact_G_Upstream, // G uncontacted G (B, n_head, seq_len, head_dim)
            model_paramaters.dV,                   // dv
            batch_size,                            // batch_size
            num_heads,                             // num_heads
            seq_len,                               // seq_len
            head_dim                               // head_dim
        );

        dp(
            model_paramaters.V_T_device_out,       // PT
            model_paramaters.Uncontact_G_Upstream, // G uncontacted G (B, n_head, seq_len, head_dim)
            model_paramaters.dP,                   // dv
            batch_size,                            // batch_size
            num_heads,                             // num_heads
            seq_len,                               // seq_len
            head_dim                               // head_dim
        );

        /*
            It's been a while not, been through a lot so. Just a small re-cap

            J(P1) = diag(P1) - P1^T P1  where P1 is exactly the one row of P = softmax(s) matrix and P1 diag is [seq_len, seq_len] dimension matrix,
            and at the end we multiply that d(scores)_row = J(P1) @ dP_row is I am not wrong ofcourse.
        */

        if (debug)
            pyDebuggerReleaseStage5();
        // Note: the bug is here since the kerenl looks fine after I unit tested it on collab.
        softmaxBackGrad(
            model_paramaters.attention_head.P,
            model_paramaters.dP,
            model_paramaters.P_T_device_out, // (B, num_head, T, T) buffer re-used
            batch_size * num_heads * seq_len * seq_len,
            batch_size,
            seq_len,
            num_heads);
        // Just to alias it properly for readability
        d_scores_device = model_paramaters.P_T_device_out;

        if (debug)
            pyDebuggerReleaseStage6();
    }
};