//
// Created by avash on 11/18/26.
//

#include "linear.hpp"


Linear::Linear(const int feature_in, const int feature_out, const int seq_len, const int batch_size, const int n_head, const bool debug)
    : f_in(feature_in), f_out(feature_out), n_head(n_head), seq_len(seq_len), batch_size(batch_size), debug(debug)
{
    this->head_dim = feature_out / n_head;
    this->LinearParams(feature_in, feature_out);
    weighted_sum_device = make_unique<Tensor<float>>(vector<int>{batch_size, seq_len, f_out}, DEVICE);
    multi_head_out_device = make_unique<Tensor<float>>(vector<int>{batch_size, seq_len, n_head, head_dim}, DEVICE);

    // In order to transpose that Q,K,V and swap we need to allocate the memory in video memory
    deviceArrInTranspose = make_unique<Tensor<float>>(vector<int>{batch_size, seq_len, n_head, head_dim}, DEVICE);
}

void Linear::LinearParams(const int fan_in, const int fan_out) {

    // To improve the kernel performance this is pre-init and then used in the second kernel.
    cudaMalloc(&d_state_weight, fan_in * fan_out * sizeof(curandState));
    cudaMalloc(&d_state_bias, fan_out * sizeof(curandState));

    device_weight = make_unique<Tensor<float>>(vector<int>{fan_in, fan_out}, DEVICE);
    device_bias = make_unique<Tensor<float>>(vector<int>{fan_out}, DEVICE);

    // making sure these seeds are different
    const unsigned long seed_1 = static_cast<unsigned long>(
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count());

    const unsigned long seed_2 = static_cast<unsigned long>(
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count());

    // we need tow kernal launches here
    KaimingInit(device_weight->get(), d_state_weight, fan_in, fan_out, seed_1);
    KaimingInit(device_bias->get(), d_state_bias, 1, fan_out, seed_2);
}

/**
 * @brief splits channels and reorders (B,T,C) into (B,H,T,D)
 *
 * @warning Returns device pointer
 * @return (batch_size, num_head, seq_len, head_dim) tensor in device
 */
float* Linear::reshapeHead() const {
    multiHeadedAttention(weighted_sum_device->get(), multi_head_out_device->get(), n_head, head_dim, batch_size, f_out, seq_len);
    return multi_head_out_device->get();
}

/**
 *
 * @class forward
 * @brief expects BTC tensor for Linear transformation
 *
 * @warning  Expects a device pointer
 * @return (Linear transformed tensor)
 */
float *Linear::forward(float *val) const {
    WeightedSum(val, device_weight->get(), device_bias->get(), weighted_sum_device->get(), batch_size * seq_len, f_in, f_out);
    return weighted_sum_device->get();
}

/**
 *
 * @class forward
 * @brief Expects a BTC tensor for the linear transformation.
 * Applies the linear transformation using the existing weights and biases
 * while supporting different batch sizes.
 *
 * If this were a high-performance API, this approach would be inefficient.
 * However, it is sufficient for our current use case.
 *
 * @param val (B, T, C) tensor
 * @param b
 * @param prompt_seq_len
 * @warning  Expects a device pointer
 * @return (Linear transformed tensor)
 */
float *Linear::forward(float *val, const int b, const int prompt_seq_len) {

    const int t = prompt_seq_len;

    weightedSumP = make_unique<Tensor<float>>(vector<int>{batch_size, seq_len, f_out}, DEVICE);
    WeightedSum(val, device_weight->get(), device_bias->get(), weightedSumP->get(), b * t, f_in, f_out);

    return weightedSumP->get();
}

/**
 * @brief splits channels and reorders (B,T,C) into (B,H,T,D)
 *
 * @param B batch size
 * @param prompt_seq_len
 * @warning Returns device pointer
 * @return (batch_size, num_head, seq_len, head_dim) tensor in device
 */
float* Linear::reshapeHead(const int B, const int prompt_seq_len) {
    const int T = prompt_seq_len;
    multiHeadedP = make_unique<Tensor<float>>(vector<int>{batch_size, seq_len, head_dim, n_head}, DEVICE);

    multiHeadedAttention(weightedSumP->get(), multiHeadedP->get(), n_head, head_dim, B, f_out, T);

    return multiHeadedP->get();
}

float* Linear::transposeMiddle4DTensor() const {
    TransposeMiddleTensor4(
    multi_head_out_device->get(),
    deviceArrInTranspose->get(),
    batch_size,
    seq_len,
    n_head,
    head_dim);

    return deviceArrInTranspose->get(); // (batch_size, num_head, seq_len, head_dim) changed shape
}

float* Linear::transposeMiddle4DTensor(const int B, const int prompt_seq_len)  {
    const int T = prompt_seq_len;
    transposedP = make_unique<Tensor<float>>(vector<int>{batch_size, seq_len, head_dim, n_head}, DEVICE);
    TransposeMiddleTensor4(
        multiHeadedP->get(),
        transposedP->get(),
        B,
        T,
        n_head,
        head_dim);
    return transposedP->get(); // (batch_size, num_head, seq_len, head_dim) changed shape
}

void Linear::transposeKeyForAttnScore(float *buffer_out_d, const int b, const int prompt_seq_len) const {
    TransposeKey(
    multiHeadedP->get(),
    buffer_out_d,
    n_head,
    head_dim,
    b,
    prompt_seq_len,
    false);
}

/**
 * @class transposeKeyForAttnScore
 * @brief transforms (batch_size, num_head, seq_len, head_dim) to (batch_size, num_head, head_dim, seq_len)
 *
 * @note Writes (batch_size, num_head, head_dim, seq_len) into buffer_out_d.
 */
void Linear::transposeKeyForAttnScore(float *buffer_out_d) const {
    TransposeKey(
    multi_head_out_device->get(),
    buffer_out_d,
    n_head,
    head_dim,
    batch_size,
    seq_len,
    false);
}

Linear::~Linear() {
    cudaFree(d_state_weight);
    cudaFree(d_state_bias);
}