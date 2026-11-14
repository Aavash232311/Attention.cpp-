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
    int N;

    Tensor(const vector<int> &shape, Location location, bool zero_init = false);
    T *get();
    
    operator T *()
    {
        return tensor;
    }
    ~Tensor();
};
