#pragma once
#include "../include/utils.hpp"
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>


extern "C" void layerNormalization(float *x, float *gamma, float *beta, int batch_size, int seq_len, int d_model);

class LayerNorm
{
    // we can loose the symmantic meaning in the norm procress so gamma and beta as learnable parms adjusts accordingly.
    float *h_gamma = nullptr;
    float *h_beta = nullptr;

    float *d_gamma;
    float *d_beta;

    float *d_x;

    int batch_size;
    int seq_len;
    int d_model;

public:
    LayerNorm(int batch_size, int seq_len, int d_model) // LayerNorm(x) = γ . (x - μ) / √(σ² + ε) + β
    {
        h_gamma = (float *)malloc(d_model * sizeof(float));
        h_beta = (float *)malloc(d_model * sizeof(float));

        this->batch_size = batch_size;
        this->seq_len = seq_len;
        this->d_model = d_model;

        for (int i = 0; i < d_model; ++i)
        {
            h_gamma[i] = 1.0f; // because these are the learnable paramaters
            h_beta[i] = 0.0f;
        }

        // allocate memory for the device
        cudaMalloc((void **)&d_gamma, d_model * sizeof(float));
        cudaMalloc((void **)&d_beta, d_model * sizeof(float));
        cudaMalloc((void **)&d_x, batch_size * seq_len * d_model * sizeof(float));

        // copy to device for both now, should be done in constuctor to get the performace advantage.
        // H2D copy
        cudaMemcpy(d_gamma, h_gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, h_beta, d_model * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~LayerNorm()
    {
        free(h_gamma);
        free(h_beta);

        cudaFree(d_beta);
        cudaFree(d_gamma);
        cudaFree(d_x);
    }

    void forward(float *x) // after adding the embeddings we have (B, T, C) shape
    {
        // we first normalize and then pass it to the transofmer which is normal in modern transformer.
        // we also have to be very precise in cases like this because the bugs are scilent and we would never know what happened.

        // first we need to calculate the mean μ
        // second standard deviation
        // then normalize gama and beta are the learnable paramaters.

        // copy x to device
        cudaMemcpy(d_x, x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
        layerNormalization(d_x, d_gamma, d_beta, batch_size, seq_len, d_model);
        cudaMemcpy(x, d_x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    }

    float *getGamma()
    {
        return this->h_gamma; // of course these are tuneable.
    }

    float *getBetta()
    {
        return this->h_beta;
    }
};