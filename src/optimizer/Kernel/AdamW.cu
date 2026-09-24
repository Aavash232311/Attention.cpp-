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

__global__ void adamw_step(
    float *theta, // This is basically pointer to the device weight location
    float *grad,  // A tensor threa and grad needs to be of same shape
    float *m,
    float *v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t, // How many times the oprimizer has been called, one per batch
    int n)
{
    // which_bloc * block_dim + offset_thread_idx,             i+= 256 * 128 block pass the total number of allocated threads
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        float g = grad[i];
        // momentum
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        // RMS prop
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        // bias correction for momentum and velocity
        float m_hat = m[i] / (1.0f - powf(beta1, t));
        float v_hat = v[i] / (1.0f - powf(beta2, t));

        // update
        theta[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * theta[i]);
    }
}

extern "C"
{

    void AdamWSTEP(
        float *theta,
        float *grad,
        float *m,
        float *v,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        int t,
        int n)

    {
        cudaMemset(m, 0, n * sizeof(float));
        cudaMemset(v, 0, n * sizeof(float));

        
        int threads = 256;
        int blocks = 128;
        adamw_step<<<blocks, threads>>>(theta, grad, m, v, lr, beta1, beta2, eps, weight_decay, t, n);

        // no sync here because updating gradients can happen in parallel without relying on each other.
    }
}