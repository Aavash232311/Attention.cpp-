#pragma once

#include "../include/utils.hpp"
#include "../include/tensor.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <memory>
#include <chrono>

extern "C" void KaimingInit(float *arr, curandState *state, int x, int y, unsigned long seed);
extern "C" void WeightedSum(float *x, float *w, float *b, float *c, int M, int K, int N);
// in math.cu kernel -Reused logic
extern "C" void multiHeadedAttention(float *ws, float *out, int num_head, int head_dimension, int batch_size, int d_model, int seq_len);
extern "C" void TransposeKey(float *arr, float *out, int num_heads, int head_dim, int batch_size, int seq_len, bool reverse);
extern "C" void TransposeMiddleTensor4(float *arr, float *out, int batch_size, int seq_len, int num_head, int head_dim);


using namespace std;
class Linear
{
    bool dLinear = true;

    int f_in;
    int f_out;
    int n_head;
    int seq_len;
    int head_dim;
    int batch_size;

    bool debug = true;

    unique_ptr<Tensor<float>> device_bias;
    unique_ptr<Tensor<float>> device_weight;
    unique_ptr<Tensor<float>> weighted_sum_device;
    unique_ptr<Tensor<float>> multi_head_out_device;
    unique_ptr<Tensor<float>> deviceArrInTranspose;

    curandState *d_state_weight{};
    curandState *d_state_bias{};

    unique_ptr<Tensor<float>> weightedSumP;
    unique_ptr<Tensor<float>> multiHeadedP;
    unique_ptr<Tensor<float>> transposedP;

    std::unique_ptr<Utility> utils = std::make_unique<Utility>();

public:
    Linear(int feature_in, int feature_out, int seq_len, int batch_size, int n_head, bool debug);
    ~Linear();

    [[nodiscard]] float *reshapeHead() const;
    [[nodiscard]] float *transposeMiddle4DTensor() const;
    [[nodiscard]] float *getBiasDevice() const{ return this->device_bias->get();}
    [[nodiscard]] float *getWeightDevice() const {return this->device_weight->get();}
    [[nodiscard]] float *getMultiHeadedDevice() const {return this->multi_head_out_device->get();}


    float *forward(float *val) const;
    void LinearParams(int fan_in, int fan_out);
    float *reshapeHead(int B, int prompt_seq_len);
    float *forward(float *val, int b, int prompt_seq_len);
    void transposeKeyForAttnScore(float *buffer_out_d) const;
    float *transposeMiddle4DTensor(int B, int prompt_seq_len);
    void transposeKeyForAttnScore(float *buffer_out_d, int b, int prompt_seq_len) const;
};