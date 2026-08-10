#include "include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>


#include "forward/linear.hpp"
#include "include/linear.hpp"
#include "include/p_head.hpp"
#include "include/cache_in.hpp"
#include "include/cache_out.hpp"
#include "forward/interface.hpp"
#include "forward/layer_norm.hpp"
#include "forward/embeddings.hpp"
#include "include/data_loader.hpp"
#include "include/netattention.hpp"
#include "forward/attention_head.hpp"
#include "include/single_embeddings.hpp"
#include "backpropagation/interface_back.hpp"

// Again my background is beginner here with little concept from C
// Transformer are complex neural network artitecture so I will focus on
// making the things right at first rather than micro level optimization
// The goal is to learn the underlying concept we can always bring the performace up later on.

// I handtyped this again why not
// I created a monolith here never mind.

// nvcc src/attention.cpp src/kernel/math.cu -o src/bin/attention
// ./src/bin/attention

// nvcc src/main.cpp src/kernel/utils.cu src/forward/Kernel/layer_norm.cu src/forward/Kernel/embedding.cu src/forward/Kernel/linear.cu src/forward/Kernel/attention_head.cu src/forward/Kernel/interface.cu src/backpropagation/Kernel/interface_back.cu -o src/bin/attention

// backprop https://arxiv.org/pdf/2307.08691



int main()
{
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    bool debug = true;
    // #ifdef DEBUG
    //     debug = true;
    // #endif

    int d_model = 32;
    int vocab_size; // that depends upon the data that you are passing.
    int num_heads = 2;
    int batch_size = 8;
    int seq_len = 4;
    int epoch = 12;
    bool drop_last = true; // for training set this to false, if someone is serious about this email me. the cost of implementing this feature will affect everything in depth many tradeoffs

    bool hyperParamaterRelease = false;

    std::string path = "./src/data/chunk.txt";

    auto utils = std::make_unique<Utility>();

    auto textEncoderFile = std::make_unique<EncoderText>();

    textEncoderFile->loadTextChunk(path); // load that into char arr

    auto &charPool = textEncoderFile->getFileAsChar();
    auto helper = std::make_unique<Helper>(charPool);

    vocab_size = charPool.size();

    // In this case we would have release the config from our C++ script
    // and python program should be able to read it.

#ifdef PARAMS
    hyperParamaterRelease = true;
#endif

    if (hyperParamaterRelease)
    {
        // Python debugger wants hyperparams to initlize item to totally remove randomness
        // to redice the kernels, if compiled with this flag then the program ends here

        releaseHyperParamaters(
            d_model,
            vocab_size,
            batch_size,
            seq_len,
            num_heads); // write the hyperparamaters down

        return 0;
    }

    // we just need a simple shample.
    epoch = debug == true ? 1 : epoch;

    const std::vector<int> &encodedData = helper->getEncodedList();

    std::unique_ptr<AttentionInterface> model = std::make_unique<AttentionInterface>(
        d_model,
        num_heads,
        batch_size,
        seq_len,
        drop_last,
        encodedData,
        debug);

    model->train(epoch);

    auto end = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize(); // CPU is waiting for the GPU to finish
    std::chrono::duration<double, std::milli> duration = end - start;

    if (!(debug))
    {
        std::cout << "Total time C++ execution (with sync overhead): " << duration.count() << " ms\n";
    }

    return 0;
}
// I wont let this project die