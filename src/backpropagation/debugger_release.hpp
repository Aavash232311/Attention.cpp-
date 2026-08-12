#pragma once
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./interface_back.hpp"

#include "../include/utils.hpp"
#include "../include/linear.hpp"
#include "../include/p_head.hpp"
#include "../include/cache_in.hpp"
#include "../include/cache_out.hpp"
#include "../include/netattention.hpp"
#include "../include/attention_params.hpp"
#include "../include/single_embeddings.hpp"

// Releases debugger script for python project to check
// Autograd class and its child classes are getting messy

class AutogradEngineDebuggerRelease : public AutoGradEngine
{
public:
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


    
};