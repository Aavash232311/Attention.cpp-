#include "attention_params.hpp"
#include "linear.hpp"
#pragma once

struct NetAttentionParamaters
{
    AttentionParamaters attention_head;
    LinearParams lm_head;

    float *L; // loss from the cross entropy loss starting of the backpropagation
    // so the above struct contains a pointer reference for CPU memory but we need to make a buffer for gpu memory right here.

    // --------- Variables needed for upstream gradient --------------


    // MAKE SURE THAT THSE ARE VARIABLES FROM THE DEVICE
    float *y_actual;
    float *y_predicted;
    float *dl_dz_out_device;
    float *dl_dz_out_host;
};
