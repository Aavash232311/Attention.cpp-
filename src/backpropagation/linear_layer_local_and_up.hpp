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
extern "C" void wt_upstream(float *w, float *wt, int d_model, int vocab_size);

class LinearLayerBackpropagation
{

public:
    int batch_size;
    int seq_len;
    int d_model;
    int vocab_size;
    bool debug;

    LinearLayerBackpropagation(
        int batch_size,
        int seq_len,
        int d_model,
        int vocab_size,
        bool debug)
    {
        this->batch_size = batch_size;
        this->seq_len = seq_len;
        this->d_model = d_model;
        this->vocab_size = vocab_size;
        this->debug = debug;
    }

    /**
     * @class LinearLayerBackpropagation
     * @brief backward does the backpropagation for local and upstream gradient in linear layer.
     *
     * @param x The input x, that gets feeded into the linear layer
     * @param G The upstream gradient that gets in the Linear
     * @param W The weight that needs to be updated
     * @param WT The placeholder for the WT to stay
     * @param dw The placeholder for the local gradient dW
     * @param db The placeholder for the bias to be updated
     * @param dx The placeholder for the upstrema gradient.
     *
     * @warning Important Node: The size of these weight might change here.
            For example in the QKV is (C, C) it may differ make sure to pass the right arguement.

     *@warning Important limitation. I made this only for lm_head and all the QKV linear with shape (C, C). If something custom arguement has to be passed between these kernels then this wont work.
     So this helps in reducing number of lines of codes and mistakes.
     */
    void backward(
        float *x,
        float *G,
        float *W,
        float *Wt,
        float *xt,
        float *dw, // local gradient W
        float *db, // local gradient bias
        float *dx,  // local gradient x
        bool special_case=false
    )
    {
        // Considering a linear layer: z = wx + b;

        // First step transpose x
        lm_head_transpose_h(
            x,
            xt,
            batch_size,
            seq_len,
            d_model);

        // These are not in sync, one does not rely on another.
        dl_dw_upstream(
            xt,
            G,
            dw,
            batch_size,
            seq_len,
            d_model,
            vocab_size);

    

        dbias(
            G,
            db,
            batch_size,
            seq_len,
            special_case ? vocab_size: d_model,
            false);

        wt_upstream(
            W,
            Wt,
            d_model,
            vocab_size);

        // upstream gradient that flows up
        dl_dh_upstream(G,
                       Wt,
                       dx, // upstream graidnet dx
                       batch_size,
                       seq_len,
                       d_model,
                       special_case ? vocab_size: d_model, 
                       true);

        
    }
};