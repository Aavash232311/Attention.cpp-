#pragma once
#include <vector>
#include <memory>
#include <random>
#include <cstdio>
#include <math.h>
#include <ranges>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <unordered_map>

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
    /**
     * @class showHashMap
     * @brief prints hash map ket and value
     *
     * @param encoded_input: pass in the hash map
     */
    void showHashMap(std::vector<std::unordered_map<char, int>> &encoded_input)
    {
        for (size_t i = 0; i < encoded_input.size(); i++)
        {
            for (const auto &pair : encoded_input[i])
            {
                std::cout << "  Key: " << pair.first << " -> Value: " << pair.second << std::endl;
            }
        }
        return;
    }

    /**
     * @class showVector
     * @brief Prints 1D vector data structure
     *
     * @param arr: pass in the vector
     */

    template <typename T>
    void showVector(std::vector<T> &arr)
    {
        for (size_t i = 0; i < arr.size(); i++)
        {
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }

    /**
     * @class Print2DVector
     * @brief Prints 2D vector data structure
     *
     * @param vec: pass in the vector
     * @param shape: bool shows the shape of the vector as if it was a tensor
     */

    template <typename T>
    void Print2DVector(const std::vector<std::vector<T>> &vec, bool shape_only = false)
    {
        if (shape_only == false)
        {
            for (const auto &row : vec)
            {
                for (const auto &value : row)
                {
                    std::cout << value << " ";
                }

                std::cout << '\n';
            }
        }
        std::string shapeAnnotation = shape_only == false ? "\n] Shape: (" : "Shape: (";

        std::cout << shapeAnnotation
                  << vec.size()
                  << ", "
                  << (vec.empty() ? 0 : vec[0].size())
                  << ")\n";
    }

    /**
     * @class Print3DVector
     * @brief Prints 3D vector data structure
     *
     * @param vec: pass in the vector
     * @param shape: bool shows the shape of the vector as if it was a tensor
     */

    template <typename T>
    void Print3DVector(const std::vector<std::vector<std::vector<T>>> &vec, bool shape_only = false)
    {
        if (!shape_only)
        {
            for (const auto &matrix : vec)
            {
                for (const auto &row : matrix)
                {
                    for (const auto &value : row)
                    {
                        std::cout << value << " ";
                    }

                    std::cout << '\n';
                }

                std::cout << '\n';
            }
        }

        std::string shapeAnnotation =
            shape_only ? "Shape: (" : "\nShape: (";

        std::cout << shapeAnnotation
                  << vec.size()
                  << ", "
                  << (vec.empty() ? 0 : vec[0].size())
                  << ", "
                  << ((vec.empty() || vec[0].empty()) ? 0 : vec[0][0].size())
                  << ")\n";
    }

    /**
     * @class print_vector
     * @brief Prints vector data structure
     *
     * @param vec: pass in the vector
     *
     */

    template <typename T>
    void print_vector(const std::vector<T> &vec)
    {
        std::cout << "[";
        for (int i = 0; i < vec.size(); i++)
        {
            std::cout << vec[i];
            std::cout << ", ";
        }
        std::cout << "]\n";
    }

    /**
     * @class printFlatArray2D
     * @brief Prints flat memory strip as if it was a tensor
     *
     * @param generic pointer address
     * @param dim 1
     * @param dim 2
     *
     */

    template <typename T>
    void printFlatArray2D(const T *arr, int seq_len, int d_model)
    {
        const int M = 16;

        printf("tensor([\n");
        for (int r = 0; r < seq_len; r++)
        {
            if (r == M && seq_len > M * 2)
            {
                printf("  ...,\n");
                r = seq_len - M;
            }
            printf("  [");
            for (int c = 0; c < d_model; c++)
            {
                if (c == M && d_model > M * 2)
                {
                    printf("..., ");
                    c = d_model - M;
                }
                printf("%g%s", (double)arr[r * d_model + c], c < d_model - 1 ? ", " : "");
            }
            printf("],\n");
        }
        printf("], shape=[%d, %d])\n", seq_len, d_model);
    }

    /**
     * @class printFlatArray3D
     * @brief Prints flat memory strip as if it was a tensor
     *
     * @param generic pointer address
     * @param dim 1
     * @param dim 2
     * @param dim 3
     *
     * @param show_last_dim bool shows the last dim
     *
     * @note Known issue something is wrong here, I noticed that when using this it sometimes crashes next time that happens
     * I will take a look at this one right here.
     */

    template <typename T>
    void printFlatArray3D(const T *arr, int seq_len, int batch_size, int embed_dim, bool show_last_dim = false)
    {
        const int M = 1024;

        printf("tensor([\n");
        for (int r = 0; r < seq_len; r++)
        {
            if (r == M && seq_len > M * 2)
            {
                printf("  ...,\n");
                r = seq_len - M;
            }
            printf("  [\n");
            for (int c = 0; c < batch_size; c++)
            {
                if (c == M && batch_size > M * 2)
                {
                    printf("    ...,\n");
                    c = batch_size - M;
                }
                printf("    [");
                for (int e = 0; e < embed_dim; e++)
                {
                    if (e == M && embed_dim > M * 2)
                    {
                        printf("..., ");
                        e = embed_dim - M;
                    }
                    printf("%g%s", (double)arr[r * batch_size * embed_dim + c * embed_dim + e], e < embed_dim - 1 ? ", " : "");
                }
                printf("],\n");
            }
            printf("  ],\n");
        }
        printf("], shape=[%d, %d, %d])\n", seq_len, batch_size, embed_dim);

        if (show_last_dim)
        {
            printf("\nlast_dim (embed_dim=%d):\n", embed_dim);
            for (int r = 0; r < seq_len; r++)
            {
                if (r == M && seq_len > M * 2)
                {
                    printf("  ...\n");
                    r = seq_len - M;
                }
                for (int c = 0; c < batch_size; c++)
                {
                    if (c == M && batch_size > M * 2)
                    {
                        printf("  ...\n");
                        c = batch_size - M;
                    }
                    printf("  [%d][%d] -> [", r, c);
                    for (int e = 0; e < embed_dim; e++)
                    {
                        if (e == M && embed_dim > M * 2)
                        {
                            printf("..., ");
                            e = embed_dim - M;
                        }
                        printf("%g%s", (double)arr[r * batch_size * embed_dim + c * embed_dim + e], e < embed_dim - 1 ? ", " : "");
                    }
                    printf("]\n");
                }
            }
        }
    }

    /**
     * @class flatArrToVec
     * @brief converts the memory strip to a vector data structure
     *
     * @param arr pointer to the first memory address
     * @param rows 1
     * @param cols 2
     *
     *
     * @note this is expensive when invoking under each epoch. Its sutiable only for a single shot opreation
     * Example: when converting flat memory strip to a vector using data loader.
     */
    template <typename T>
    std::vector<std::vector<T>> flatArrToVec(const float *arr, int rows, int cols)
    {
        std::vector<std::vector<T>> vec2d(rows, std::vector<T>(cols)); // rows and cols allocation

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                vec2d[r][c] = static_cast<T>(arr[r * cols + c]);
            }
        }

        return vec2d;
    }

    /**
     * @class TwoDVectorToFlatMem
     * @brief converts vector data structures to a flat memory strip
     *
     * @param arr pass in vector data structure
     * @param shape prints the shape
     *
     * @warning This is again expensive task to invoke under each epoch. It really hurts the performance very much.
     * Use this in something like a data loader for single shot opreation
     */
    template <typename T>
    T *TwoDVectorToFlatMem(const std::vector<std::vector<T>> &arr,
                           bool shape = false) // This is a bottleneck in the perforamce I know the fact that we should use flat vector but for the sake of learning I am using this.
    {
        if (arr.empty())
            return nullptr;

        size_t rows = arr.size();
        size_t cols = arr[0].size();

        // std::cout << "Rows: " << rows << " Cols: " << cols << std::endl;

        for (const auto &row : arr)
        {
            if (row.size() != cols)
            {
                throw std::runtime_error(
                    "TwoDVectorToFlatMem: jagged vectors are not supported.");
            }
        }

        T *newArr = new T[rows * cols];

        if (shape)
        {
            std::cout << "Shape: ("
                      << rows << ", "
                      << cols << ")\n";
        }

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                newArr[r * cols + c] = arr[r][c];
            }
        }

        return newArr;
    }

    // This is to debug the last mtrices which are multiplied or masked, since its very hard to make a judgement on a 4D tensor.
    // void print2DMatrixLastTwo(
    //     float *arr,
    //     int batch_size,
    //     int n_head,
    //     int seq_len,
    //     const char *label = "Matrix")
    // {
    //     for (int b = 0; b < batch_size; b++)
    //     {
    //         for (int h = 0; h < n_head; h++)
    //         {
    //             printf("%s [%d][%d]\n", label, b, h);
    //             for (int row = 0; row < seq_len; row++)
    //             {
    //                 printf("  [ ");
    //                 for (int col = 0; col < seq_len; col++)
    //                 {
    //                     int idx = b * (n_head * seq_len * seq_len) + h * (seq_len * seq_len) + row * (seq_len) + col;
    //                     printf("%10.4f  ", arr[idx]);
    //                 }
    //                 printf("]\n");
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    /**
     * @class print2DMatrixLastTwo
     * @brief Prints 4d memory strip's last two matrix.
     *
     * @param arr pass in vector data structure
     * @param dim1
     * @param dim2
     * @param dim3
     * @param dim4
     *
     * @brief Used for debugging
     */

    void print2DMatrixLastTwo(
        float *arr,
        int batch_size,
        int n_head,
        int dim_row,
        int dim_col)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int h = 0; h < n_head; h++)
            {
                printf("%s [%d][%d]\n", "Matrix", b, h);
                for (int row = 0; row < dim_row; row++)
                {
                    printf("  [ ");
                    for (int col = 0; col < dim_col; col++)
                    {
                        int idx = b * (n_head * dim_row * dim_col) + h * (dim_row * dim_col) + row * dim_col + col;
                        printf("%10.4f  ", arr[idx]);
                    }
                    printf("]\n");
                }
                printf("\n");
            }
        }
    }

    /**
     * @class print2DMatrixLastTwoRect
     * @brief Prints 4d memory strip's last two matrix.
     *
     * @param arr pass in vector data structure
     * @param dim1
     * @param dim2
     * @param dim3
     * @param dim4
     * @param label: can pass in anything for label
     *
     * @brief Used for debugging
     */

    // This the more flexible. used for checking dimension in tensor that are swapped.
    void print2DMatrixLastTwoRect(
        float *arr,
        int batch_size,
        int n_head,
        int seq_len,
        int head_dim,
        const char *label = "Matrix")
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int h = 0; h < n_head; h++)
            {
                printf("%s [%d][%d]\n", label, b, h);
                for (int row = 0; row < seq_len; row++)
                {
                    printf("  [ ");
                    for (int col = 0; col < head_dim; col++)
                    {
                        int idx = b * (n_head * seq_len * head_dim) + h * (seq_len * head_dim) + row * (head_dim) + col;
                        printf("%10.4f  ", arr[idx]);
                    }
                    printf("]\n");
                }
                printf("\n");
            }
        }
    }

    /**
     * @class printFlarArray4D
     * @brief Prints the 4d memory strip as if it was a tensor
     *
     * @param data pointer data
     * @param dim1
     * @param dim2
     * @param dim3
     * @param dim4
     *
     * @brief Used for debugging
     */
    template <typename T>
    void printFlarArray4D(T *data, int B, int S, int num_heads, int head_dim)
    {
        std::cout << "shape: (" << B << ", " << S << ", " << num_heads << ", " << head_dim << ")\n";
        std::cout << "total elements: " << B * S * num_heads * head_dim << "\n\n";

        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                for (int h = 0; h < num_heads; h++)
                {
                    std::cout << "  [" << b << "][" << s << "][" << h << "] | ";
                    for (int d = 0; d < head_dim; d++)
                    {
                        int idx = (b * S + s) * (num_heads * head_dim) + h * head_dim + d;
                        std::cout << "[" << idx << "]=" << data[idx] << " ";
                    }
                    std::cout << "\n";
                }
    }

    /*
        So I cannot make judgement in 4D tensor which is printed as flat.
        Espically for swapNT I want to verify that claim using this method
    */

    void print2DTensorOnDemmand(float *data, int dim0, int dim1, int dim2, int dim3,
                                int fix0, int fix1)
    {
        int base = fix0 * (dim1 * dim2 * dim3) + fix1 * (dim2 * dim3);
        printf("\n[%d][%d][*][*] (%d x %d):\n", fix0, fix1, dim2, dim3);
        for (int i = 0; i < dim2; i++)
        {
            for (int j = 0; j < dim3; j++)
                printf("%8.3f ", data[base + i * dim3 + j]);
            printf("\n");
        }
    }

    template <typename T>
    void printLastOneOf3D(const T *arr, int D0, int D1, int D2)
    {
        int i = D0 - 1; // last batch
        int j = D1 - 1; // last row

        std::cout << "[";
        for (int k = 0; k < D2; ++k)
        {
            size_t idx = i * D1 * D2 + j * D2 + k;
            std::cout << arr[idx];
            if (k + 1 < D2)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // THIS IS FOR PRINTHING ALL THE ROWS OF ONE HOT ENCODED ARRAY

    template <typename T>
    void printAllOneHot3D(const T *arr, int D0, int D1, int D2)
    {
        for (int i = 0; i < D0; ++i)
        {
            for (int j = 0; j < D1; ++j)
            {
                std::cout << "[" << i << "," << j << "]: [";
                for (int k = 0; k < D2; ++k)
                {
                    size_t idx = i * D1 * D2 + j * D2 + k;
                    std::cout << arr[idx];
                    if (k + 1 < D2)
                        std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }
    }

    // THIS IS FOR PRINTING FLAT ARRAY OUTPUT
    // If type is device we allocate temp memory to print
    template <typename T>
    void printFlatArray1D(const T *arr, int N)
    {
        int max_print = N;
        std::cout << "tensor([";
        for (int i = 0; i < N; ++i)
        {
            if (N > max_print && i == max_print / 2)
            {
                std::cout << "..., ";
                i = N - max_print / 2 - 1; // skip to near end
                continue;
            }
            std::cout << std::fixed << std::setprecision(5) << arr[i];
            if (i + 1 < N)
                std::cout << ", ";
        }
        std::cout << "], shape=[" << N << "])\n";
    }
};
