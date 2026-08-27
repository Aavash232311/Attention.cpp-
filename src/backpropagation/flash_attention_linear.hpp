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

class FlashAttentionLinear : virtual public AutoGradEngine
{
public:
    FlashAttentionLinear(int d_model, int vocab_size, int num_heads,
                         int seq_len, int batch_size, bool debug)
        : AutoGradEngine(d_model, vocab_size, num_heads, seq_len, batch_size, debug)
    {
    }


    // G_kx0 total upstream gradient and Linear Layer, add-residual back propagation here.
    void invoke()
    {
        if (debug)
        {
            cout << "Invoked method" << endl;
        }
    }
};