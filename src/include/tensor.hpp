#pragma once

#include <iosfwd>
#include <vector>
#include <memory>
#include <random>
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
    int N;
    Location location;

    Tensor(vector<int> shape, Location location);
    int friend intVecSum(vector<int> vec);

    template <typename U>
    friend ostream &operator<<(ostream &out, const Tensor<U> &t);
    ~Tensor();
};
