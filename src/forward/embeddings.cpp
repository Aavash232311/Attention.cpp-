//
// Created by avash on 11/18/26.
//

#include "embeddings.hpp"


Embeddings::Embeddings(const int d_model, const int vocab_size, const int seq_len, const int batch_size, const bool debug)
    : d_model(d_model), vocab_size(vocab_size), seq_len(seq_len), batch_size(batch_size), debug(debug)
{
    cudaMalloc((void **)&devicePositionalEncoding, seq_len * d_model * sizeof(float));
    positionalEmbeddings(devicePositionalEncoding, seq_len, d_model); // Let positional embedding stay on the global memory

    this->tokenEmbeddings = init->EmbeddingInit(this->vocab_size, this->d_model); // token embeddings, Shape(vocab_size, d_model)
    // initialize fixed memory here so that our program runs faster.
    hostEmbeddings = utils->TwoDVectorToFlatMem(this->tokenEmbeddings);                 // this is from I am learning so I was too scared to touch kernel

    const size_t sizeFinalEmbeddings = seq_len * d_model * batch_size * sizeof(float);

    lookedUpEmbeddings = make_unique<Tensor<float>>(vector<int>(sizeFinalEmbeddings), HOST);
    // copy to GPU memory
    // copy embeddings to the GPU after making int flat
    cudaMalloc((void **)&deviceEmbeddings, vocab_size * d_model * sizeof(float));
    cudaMemcpy(deviceEmbeddings, hostEmbeddings, vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice);

    // allocare GPU memory for final embeddings
    cudaMalloc((void **)&deviceFinalEmbeddings, seq_len * d_model * batch_size * sizeof(float));
    // allocate memory for output after lookup Shape(seq_len, batch_size, d_model)
    cudaMalloc((void **)&lookupDeviceOut, seq_len * batch_size * d_model * sizeof(float));
}

void Embeddings::freePromptBuffers() {
    cudaFree(deviceXPrompt);
    deviceXPrompt = nullptr;
    cudaFree(lookupDeviceOutPrompt);
    lookupDeviceOutPrompt = nullptr;
    cudaFree(deviceEmbeddingsFinalPrompt);
    deviceEmbeddingsFinalPrompt = nullptr;
}

/**
 * @class Embeddings
 * @brief Embedding forward, token embeddings and positional encoding.
 *
 *
 * @param currentBatch batch (B, T) from the host.
 * @warning expects the batch in GPU
 *
 * @warning  returns BTC tensor from the device
 * @returns returns BTC tensor from the device
 * @author Avash Lamichhane
 *
 */
float *Embeddings::forward(const Batch &currentBatch)
{
    const int actual_batch = batch_size;
    lookup(currentBatch.x, deviceEmbeddings, lookupDeviceOut, d_model, seq_len, actual_batch, vocab_size); // actual batch because the batch size may vary here
    addEmbeddings(lookupDeviceOut, devicePositionalEncoding, deviceFinalEmbeddings, d_model, seq_len, actual_batch);
    debug = false;
    return deviceFinalEmbeddings;
}

float* Embeddings::forward(const int *prompt, const int B, const int prompt_seq_len) {
    freePromptBuffers();
    // utils->printFlatArray2D(deviceEmbeddings, vocab_size, d_model, "device");
    cudaMalloc(&deviceXPrompt, B * prompt_seq_len * sizeof(int));
    cudaMalloc(&lookupDeviceOutPrompt, B * prompt_seq_len * d_model * sizeof(float));
    cudaMalloc(&deviceEmbeddingsFinalPrompt, B * prompt_seq_len * d_model * sizeof(float));

    // Inference callers pass the GPU buffer returned by vectorToFlatStrip(..., "gpu").
    cudaMemcpy(deviceXPrompt, prompt, B * prompt_seq_len * sizeof(int), cudaMemcpyDeviceToDevice);

    lookup(deviceXPrompt, deviceEmbeddings, lookupDeviceOutPrompt, d_model, prompt_seq_len, B, vocab_size);
    addEmbeddings(lookupDeviceOutPrompt, devicePositionalEncoding, deviceEmbeddingsFinalPrompt, d_model, prompt_seq_len, B);

    return deviceEmbeddingsFinalPrompt;
}

Embeddings::~Embeddings() {
    (deviceXPrompt != nullptr ? cudaFree(deviceXPrompt) : cudaSuccess);
    (lookupDeviceOutPrompt != nullptr ? cudaFree(lookupDeviceOutPrompt) : cudaSuccess);
    (deviceEmbeddingsFinalPrompt != nullptr ? cudaFree(deviceEmbeddingsFinalPrompt) : cudaSuccess);

    delete[] hostEmbeddings;


    cudaFree(deviceEmbeddings);
    cudaFree(deviceFinalEmbeddings);

    cudaFree(devicePositionalEncoding);
    cudaFree(lookupDeviceOut);
}
