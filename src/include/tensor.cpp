#include "./tensor.hpp"

static size_t sizeElem(const vector<int> &shape)
{
    size_t count = 1;
    for (size_t i = 0; i < shape.size(); ++i)
        count *= shape[i];
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

template class Tensor<float>;
// I am giving opreator overloading a pause here for now