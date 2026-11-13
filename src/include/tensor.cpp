#include "./tensor.hpp"


template <typename T>
ostream &operator<<(ostream &out, const Tensor<T> &t)
{
    
    return out;
}


int intVecSum(vector<int> vec)
{
    int N = 0;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        N += vec[i];
    }
    return N;
}


template <typename T>
Tensor<T>::Tensor(vector<int> shape, Location location)
    : shape(shape), location(location)
{

    size_t N = 1;
    for (int dimension : shape)
        N *= dimension;

    if (this->location == DEVICE)
    {
        cudaMalloc((&tensor), N * sizeof(T));
    }
    if (this->location == HOST)
    {
        tensor = (T *)malloc(N * sizeof(T));
    }

    std::cout << N << std::endl;

    this->N = N;
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


template class Tensor<float>;
template ostream &operator<< <float>(ostream &, const Tensor<float> &);
