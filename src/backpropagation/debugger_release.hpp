#pragma once
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./interface_back.hpp"
#include "./flash_attention.hpp"

#include "../include/utils.hpp"
#include "../include/linear.hpp"
#include "../include/p_head.hpp"
#include "../include/cache_in.hpp"
#include "../include/cache_out.hpp"
#include "../include/netattention.hpp"
#include "../include/attention_params.hpp"
#include "../include/single_embeddings.hpp"

using namespace std;

// Releases debugger script for python project to check
// Autograd class and its child classes are getting messy

class AutogradEngineDebuggerRelease : public FlashAttention
{
public:
    AutogradEngineDebuggerRelease(int d_model, int vocab_size, int num_heads,
                                  int seq_len, int batch_size, bool debug)
        : FlashAttention(d_model, vocab_size, num_heads, seq_len, batch_size, debug)
    {
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

        // if (debug)
        // {
        //     cout << "delta" << endl;
        //     utils->printFlatArray3D(model_paramaters.dl_dz_out_host, batch_size,
        //     seq_len, vocab_size);
        // }

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
     * @brief Releases paramaters w^t, w and dl_dh (Which is the upstream gradient G)
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Can call this in any order because w is independent of the result from pervious things like h.
     */

    void pyDebuggerReleaseStage3()
    {
        // copy w^T to host for the python script to read the binary

        float *wt_host = (float *)malloc(d_model * vocab_size * sizeof(float));
        float *dl_dh_host = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));

        cudaMemcpy(wt_host, model_paramaters.wt_out_d, d_model * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dl_dh_host, model_paramaters.Contact_G_Upstream, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

        bulkRelease<float>(
            {
                {wt_host, d_model * vocab_size, "wt.bin"},
                {model_paramaters.w_host, d_model * vocab_size, "w.bin"},
                // for now this is the G shape (B, T, C)
                {dl_dh_host, batch_size * seq_len * d_model, "dl_dh.bin"},
            });

        // if (debug)
        // {
        //     cout << "W^T from C++" << endl;
        //     utils->printFlatArray2D(wt_host, vocab_size, d_model);
        // }
        free(dl_dh_host);
        free(wt_host);
    }

    /**
     * @class pyDebuggerReleaseStage4
     * @brief Releases paramaters P, V for now, reason this is a seperate method is because we might have something else to release in future.
     *
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Can call this in any order because w is independent of the result from pervious things like h.
     */

    void pyDebuggerReleaseStage4()
    {
        bulkRelease<float>(
            {
                {model_paramaters.attention_head.P, batch_size * num_heads * seq_len * seq_len, "P.bin"},
                {model_paramaters.attention_head.V, batch_size * num_heads * seq_len * head_dim, "V.bin"},
            });
    }

    /**
     * @class pyDebuggerReleaseStage5
     * @brief Releases paramaters P^T and V^T for back most layer of the flash attention, also un-contact G of 3D tensor, also dP, dV matmul
     * Also UNCONTACTS UPSTREAM GRADIENT G, from (B, T, C) to (batch_size, seq_len, num_head, head_dim)
     *
     * These methods are in sequential order, so this releases the P^T and V^T for a python debugger to verify and check
     * model_paramaters.attention_head.P and model_paramaters.attention_head.V should be consumed by the transpose Kernel.
     * @note Call this after all the 3, 2, 1 stage are released
     */
    void pyDebuggerReleaseStage5()
    {
        // Uncontact_G_Upstream is in Global mem
        float *UncG_host = (float *)malloc(batch_size * seq_len * num_heads * head_dim * sizeof(float));
        float *dP = (float *)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
        float *dV = (float *)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));

        cudaMemcpy(UncG_host, model_paramaters.Uncontact_G_Upstream, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dP, model_paramaters.dP, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dV, model_paramaters.dV, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

        bulkRelease<float>(
            {
                {model_paramaters.attention_head.P, batch_size * num_heads * seq_len * seq_len, "pt.bin"},
                {model_paramaters.attention_head.V, batch_size * num_heads * head_dim * seq_len, "vt.bin"}, // keep in mind of the transposed shape here
                {UncG_host, batch_size * seq_len * num_heads * head_dim, "G_uncontact.bin"},
                {dP, batch_size * num_heads * seq_len * seq_len, "dp.bin"},
                {dV, batch_size * num_heads * seq_len * head_dim, "dv.bin"}
            });
        free(UncG_host);
        free(dP);
        free(dV);
    }
};