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

// G_kx0 total upstream gradient and Linear Layer, add-residual back propagation here.
class FlashAttentionLinear : virtual public AutoGradEngine
{
public:
    FlashAttentionLinear(int d_model, int vocab_size, int num_heads,
                         int seq_len, int batch_size, bool debug)
        : AutoGradEngine(d_model, vocab_size, num_heads, seq_len, batch_size, debug)
    {
    }



    /*
        Shape note:

        Shape dK: (batch_size, num_heads, seq_len, head_dim)
        Shape dQ: (batch_size, num_heads, seq_len, head_dim)
        Shape dV: (batch_size, num_heads, seq_len, head_dim)

        Shape W's   : (d_model, vocab_size) 
        Shape W'ts  : (voab_size, d_model)

        We aready have the transpose kernel here. 
        Not sure to re-use that kerenl or write a new one reading in a transpose way, too old for that.


    */


    void weightTransposeAttn()
    {
        
    }


    void NormLinearNet()
    {
        //
        if (debug)
        {
            cout << "Invoked method" << endl;
        }
    }
};