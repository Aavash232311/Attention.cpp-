//
// Created by avash on 11/20/26.
//

#include "layer_norm.hpp"


void LayerNorm::forward(float *x, const int B, const int T) const {
    layerNormalization(x, d_gamma->get(), d_beta->get(), std_dev_cache->get(), mean_cache->get(), B, T, d_model);
}

void LayerNorm::forward(float *x) {
    // we first normalize and then pass it to the transfer which is normal in modern transformer.
    // we also have to be very precise in cases like this because the bugs are silent and we would never know what happened.

    // first we need to calculate the mean μ
    // second standard deviation
    // then normalize gama and beta are the learnable parameters.

    layerNormalization(x, d_gamma->get(), d_beta->get(), std_dev_cache->get(), mean_cache->get(), batch_size, seq_len, d_model);


    debug = false;
}

float* LayerNorm::getGammaDevice() const {
    return this->d_gamma->get();
}

float* LayerNorm::getStdDev() const {
    return this->std_dev_cache->get();
}

 float *LayerNorm::getMean() const{
    return this->mean_cache->get();
}

float *LayerNorm::getBeta() const {
    return this->d_beta->get();
}

LayerNorm::LayerNorm(const int batch_size, const int seq_len, const int d_model) {
    h_gamma = make_unique<Tensor<float>>(vector<int>{d_model}, HOST);
    h_beta = make_unique<Tensor<float>>(vector<int>{d_model}, HOST);

    this->batch_size = batch_size;
    this->seq_len = seq_len;
    this->d_model = d_model;

    utils = make_unique<Utility>();

    for (int i = 0; i < d_model; ++i)
    {
        h_gamma->get()[i] = 1.0f; // because these are the learnable parameters
        h_beta->get()[i] = 0.0f;
    }

    // allocate memory for the device
    d_gamma = make_unique<Tensor<float>>(vector<int>{d_model}, DEVICE);
    d_beta = make_unique<Tensor<float>>(vector<int>{d_model}, DEVICE);

    // copy to device for both now, should be done in constructor to get the performance advantage.
    // H2D copy
    cudaMemcpy(d_gamma->get(), h_gamma->get(), d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta->get(), h_beta->get(), d_model * sizeof(float), cudaMemcpyHostToDevice);


    std_dev_cache = make_unique<Tensor<float>>(vector<int>{batch_size, seq_len}, DEVICE);
    mean_cache = make_unique<Tensor<float>>(vector<int>{batch_size, seq_len}, DEVICE);
}
