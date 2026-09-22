#pragma once
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

using namespace std;

/**
 * @class AdamWPhysics
 * @brief Momentum and velocity structure for AdamW optimizer, allocate memory in GPU, sets data in float
 *
 * 
 * @param N number of float required
 * 
 * @warning Do not call this insde of each epoch
 * @author Avash Lamichhane
 *
 */
struct AdamWPhysics
{
    float* m_d;
    float* v_d;

    void init(int N)
    {
        cudaMalloc((void**)&m_d, N * sizeof(float));
        cudaMalloc((void**)&v_d, N * sizeof(float));
    }

    ~AdamWPhysics()
    {
        cudaFree(m_d);
        cudaFree(v_d);
    }
};
