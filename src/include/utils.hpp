#pragma once
#include <vector>
#include <memory>
#include <random>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <unordered_map>

using namespace std;

struct Tensor4
{
    int x;
    int y;
    int z;
    int z1; // shapes can have muliple heads
};

class Utility
{
public:
    void showHashMap(std::vector<std::unordered_map<char, int>> &encoded_input);
    template <typename T>
    void showVector(std::vector<T> &arr);

    template <typename T>
    void Print2DVector(const std::vector<std::vector<T>> &vec, bool shape_only = false);
    template <typename T>
    void Print3DVector(const std::vector<std::vector<std::vector<T>>> &vec, bool shape_only = false);

    template <typename T>
    void print_vector(const std::vector<T> &vec);

    template <typename T>
    void printFlatArray2D(const T *arr, int seq_len, int d_model, std::string pointer_type = "host");

    vector<vector<float>> lastDimHostToVector(
        const float *tensor, int A, int B, int C);

    template <typename T>
    void printFlatArray3D(const T *arr, int seq_len, int batch_size, int embed_dim, bool show_last_dim = false, std::string pointer_type = "host");

    template <typename T>
    std::vector<std::vector<T>> flatArrToVec(const float *arr, int rows, int cols);

    template <typename T>
    T *vectorToFlatStrip(std::vector<T> arr, std::string type = "cpu");

    template <typename T>
    T *TwoDVectorToFlatMem(const std::vector<std::vector<T>> &arr,
                           bool shape = false); // This is a bottleneck in the perforamce I know the fact that we should use flat vector but for the sake of learning I am using this.
    template <typename T>
    void print2DMatrixLastTwo(
        T *arr,
        int batch_size,
        int n_head,
        int dim_row,
        int dim_col,
        std::string pointer_type = "host");

    template <typename T>
    void print2DMatrixLastTwoRect(
        T *arr,
        int batch_size,
        int n_head,
        int seq_len,
        int head_dim,
        const char *label = "Matrix",
        std::string pointer_type = "host");
    template <typename T>
    void printFlarArray4D(T *data, int B, int S, int num_heads, int head_dim, std::string pointer_type = "host");

    /*
        So I cannot make judgement in 4D tensor which is printed as flat.
        Espically for swapNT I want to verify that claim using this method
    */
    void print2DTensorOnDemmand(float *data, int dim0, int dim1, int dim2, int dim3,
                                int fix0, int fix1);
    template <typename T>
    void printLastOneOf3D(const T *arr, int D0, int D1, int D2, std::string pointer_type = "host");

    // THIS IS FOR PRINTHING ALL THE ROWS OF ONE HOT ENCODED ARRAY
    template <typename T>
    void printAllOneHot3D(const T *arr, int D0, int D1, int D2);

    // THIS IS FOR PRINTING FLAT ARRAY OUTPUT
    // If type is device we allocate temp memory to print
    template <typename T>
    void printFlatArray1D(const T *arr, int N, string label = "", std::string pointer_type = "host");

    template <typename T>
    float averageFlatStrip2D(const T *arr, int N);

    template <typename T>
    void checkBuf(const char *name, T *d, size_t n);

    template <typename T>
    void tensor_min_max(T *tensor, int N, std::string type);
};
