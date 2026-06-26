#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "include/netattention.hpp"
#include "include/helper.hpp"

// The chain rule
class AutoGradEngine
{
    // welcome to my calculas class

    NetAttentionParamaters model_paramaters;

    bool debug = true;

    // ------- For utility purpose -------------
    int d_model;
    int vocab_size;
    int num_heads;
    int seq_len;
    int batch_size;
    int head_dim;

    // ---------- Handy methods -----------
    std::unique_ptr<Utility> utils = std::make_unique<Utility>();

private:
    // NOTE- TO MAKE SURE WE DO NOT RESERVE TOO MUCH MEMORY ON THE RUNETIME USE THE
    // "BUFFER" FROM THE STRUCTURE


public:
    AutoGradEngine(
        int d_model,
        int vocab_size,
        int num_heads,
        int seq_len,
        int batch_size)
    {
        this->d_model = d_model;
        this->vocab_size = vocab_size;
        this->num_heads = num_heads;
        this->seq_len = seq_len;
        this->batch_size = batch_size;

        this->head_dim = d_model / num_heads;

        // NOTE- Memory allocation in RAM or VRAM is done per epoch if done here
        // huritng the performace, allocate and re-use ones from the attention
        // consturcotr and pass as a buffer.
    }

    void backprop(
        const NetAttentionParamaters &paramaters)
    {
        this->model_paramaters = paramaters;


        if (debug)
        {
            // std::cout << "Loss " << seq_len * batch_size << std::endl;
            // utils->printFlatArray1D(paramaters.L, seq_len * batch_size);

            // std::cout << "Actual proballity" << std::endl;
            // utils->printFlatArray3D(paramaters.y_actual, batch_size, seq_len, vocab_size);

            // std::cout << "Predicted proballity" << std::endl;
            // utils->printFlatArray3D(paramaters.y_predicted, batch_size, seq_len, vocab_size);
        }

        debug = false;
    }
};
