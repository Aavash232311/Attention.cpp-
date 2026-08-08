#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./interface_back.hpp"

#pragma once

// autograd in attention head, only 1 block transformer now
// I cannot play around much when complexity grows here.
class FLashAttention: AutoGradEngine {
    // Note:- Here the upstream grad is from the Linear Layer
    // i.e lm head.


};