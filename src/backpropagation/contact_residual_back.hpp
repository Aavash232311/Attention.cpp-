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



// Contact and add residual backpropagation

class ContactResidualBack: virtual public AutoGradEngine {
    public:
       ContactResidualBack(int d_model, int vocab_size, int num_heads,
                         int seq_len, int batch_size, bool debug)
        : AutoGradEngine(d_model, vocab_size, num_heads, seq_len, batch_size, debug)
    {

    }

    void outputProj()
    {
        if (debug)
        {
            cout << "Invoked before the attention head " << endl;
        }
    }
};