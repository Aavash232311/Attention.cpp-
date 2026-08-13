#include <iostream>
#include <iterator>
#include <math.h>
#include <mma.h>
#include <random>
#include <vector>
#include <cfloat>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>


__global__ void PTG_kernel(
    float *PT,
    float *G,
    int B,
    int T,
    int C,
    int vocab_size
)
{

}


extern "C" 
{
    
}