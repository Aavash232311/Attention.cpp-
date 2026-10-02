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
extern "C" void dl_dh_upstream(float *detla, float *wt, float *out, int B, int T, int C, int vocab_size, bool sync);
extern "C" void dbias(float *G, float *dbias, int B, int T, int C, bool sync);
extern "C" void dl_dw_upstream(float *deta, float *ht, float *out, int B, int T, int C, int vocab_size);
extern "C" void lm_head_transpose_h(float *h, float *out, int B, int T, int C);

class LinearLayerBackpropagation
{

    public:
    int batch_size;
    int seq_len;
    int d_model;
    int vocab_size;

    LinearLayerBackpropagation(
        int batch_size,
        int seq_len,
        int d_model,
        int vocab_size
    )
    {
        this->batch_size = batch_size;
        this->seq_len = seq_len;
        this->d_model = d_model;
        this->vocab_size = vocab_size;
    }

    /**
     * @class LinearLayerBackpropagation
     * @brief backward does the backpropagation for local and upstream gradient in linear layer.
     *
     * @param x The input x, that gets feeded into the linear layer
     * @param G The upstream gradient that gets in the Linear
     * @param W The weight that needs to be updated
     * @param dw The placeholder for the local gradient dW
     * @param db The placeholder for the bias to be updated
     * @param dx The placeholder for the upstrema gradient.
     *
     */
    void backward(
        float *x,
        float *G,
        float *W,
        float *xt,
        float *dw, // local gradient W
        float *db, // local gradient bias
        float *dx  // local gradient x
    )
    {
        // Considering a linear layer: z = wx + b;

        // First step transpose x
        // lm_head_transpose_h(
        //     x,
        //     xt,
        //     batch_size,
        //     seq_len,
        //     d_model            
        // );

        
    }
};