#include "./tensor.hpp"


static size_t sizeElem(const vector<int> &shape)
{
    size_t count = 1;
    for (int i : shape)
        count *= i;
    return count;
}

template <typename T>
Tensor<T>::Tensor(const vector<int> &shape, Location location, bool zero_init)
    : shape(shape), location(location)
{
    size_t count = sizeElem(shape);
    if (location == DEVICE)
    {
        cudaMalloc(&tensor, count * sizeof(T));
        if (zero_init)
        {
            cudaMemset(tensor, 0, count * sizeof(T));
        }
    }
    else
    {
        tensor = (T *)malloc(count * sizeof(T));
        if (zero_init)
        {
            for (size_t i = 0; i < count; ++i)
            {
                tensor[i] = 0;
            }
        }
    }
    this->N = count;
}

template <typename T>
Tensor<T>::~Tensor()
{
    if (this->location == DEVICE)
    {
        cudaFree(tensor);
    }
    if (this->location == HOST)
    {
        free(tensor);
    }
}

template <typename T>
T *Tensor<T>::get()
{
    return this->tensor;
}

template <typename T>
Tensor<T> &Tensor<T>::NormalInit(float stdDev, float center)
{
    std::mt19937 gen(1234);

    normal_distribution<float> gaussian(center, stdDev);
    // This single kernel launch will be complicated do here
    // Its initliized once from a single instance so just create in CPU and
    // then copy to GPU.

    T *tempCPU = nullptr;
    if (location == DEVICE)
    {
        tempCPU = (T *)malloc(N * sizeof(T));
    }

    for (size_t i = 0; i < N; ++i)
    {
        if (location == DEVICE)
        {
            tempCPU[i] = gaussian(gen);
        }
        if (location == HOST)
        {
            tensor[i] = gaussian(gen);
        }
    }

    if (location == DEVICE)
    {
        // copy to tensor as GPU
        cudaMemcpy(tensor, tempCPU, N * sizeof(T), cudaMemcpyHostToDevice);
        free(tempCPU);
    }

    return *this;
}

template class Tensor<float>;
// I am giving operator overloading a pause here for now