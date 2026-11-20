#pragma once
#include "../include/utils.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <cstdio>

#include "../include/tensor.hpp"

using namespace std;

extern "C" void layerNormalization(float *x, float *gamma, float *beta, float *std_dev_cache, float *mean_cache, int batch_size, int seq_len, int d_model);

class LayerNorm
{
    // we can loose the semantic meaning in the norm progress so gamma and beta as learnable parms adjusts accordingly.
    unique_ptr<Tensor<float>> h_gamma;
    unique_ptr<Tensor<float>> h_beta;

    unique_ptr<Tensor<float>> d_gamma;
    unique_ptr<Tensor<float>> d_beta;

    int batch_size;
    int seq_len;
    int d_model;

    unique_ptr<Tensor<float>> std_dev_cache;
    unique_ptr<Tensor<float>> mean_cache;

    bool debug = true;

    unique_ptr<Utility> utils;

public:
    LayerNorm(int batch_size, int seq_len, int d_model); // LayerNorm(x) = γ . (x - μ) / √(σ² + ε) + β

    void forward(float *x); // after adding the embeddings we have (B, T, C) shape
    void forward(float *x, int B, int T) const;
    [[nodiscard]] float *getGammaDevice() const;
    [[nodiscard]] float *getStdDev() const;
    [[nodiscard]] float *getMean() const;
    [[nodiscard]] float *getBeta() const;
};