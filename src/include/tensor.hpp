#pragma once

#include <iosfwd>
#include <vector>
#include <memory>
#include <random>
#include <tuple>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

enum Location
{
    HOST,
    DEVICE
};

// Minimal tensor
template <typename T>
class Tensor
{
private:
    T *tensor = nullptr;

public:
    vector<int> shape;
    Location location;
    int N = 1;

    explicit Tensor(const vector<int> &shape, Location location, bool zero_init = false);
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    
    Tensor<T>& NormalInit(float stdDev = 0.9f, float center=0.0f);
    T *get();
    
    // The owner must stay alive while its pointer is being used.
    operator T *() &
    {
        return tensor;
    }
    operator T *() && = delete;
    ~Tensor();
};
