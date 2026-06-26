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

// I will be really honest it takes time for me to think deeply, but I will figure it out
// till this day AI hallucination is common on task that requires deep thinking.
// whatever even if this market is brutal I am investing my effort into thinking.
class Helper
{
private:
    std::vector<std::unordered_map<char, int>> encodingKeyPairs;

public:
    Helper(std::vector<char> encodingPool)
    {
        // okay so we have this array of characters based on their poistion first let's encode them.
        encodingKeyPairs.resize(encodingPool.size()); // resize

        for (size_t i = 0; i < encodingPool.size(); i++)
        {
            encodingKeyPairs[i][encodingPool[i]] = i;
        }
    }
    // getter to get the encoding pool of everything that we plan to encode on.
    std::vector<int> getEncodedList()
    {
        std::vector<int> newList;
        newList.reserve(encodingKeyPairs.size());

        for (size_t i = 0; i < encodingKeyPairs.size(); i++)
        {
            for (const auto &pair : encodingKeyPairs[i])
            {
                newList.push_back(pair.second);
            }
        }

        return newList;
    }

    std::vector<int> encoder(std::string &input_text)
    {
        // let's iterate on what we have and we will check for the encoded dict.
        std::vector<int> indices;
        indices.reserve(input_text.length());

        for (size_t i = 0; i < input_text.length(); i++)
        {
            // let's just do a linear search here. It's just a encoder/decoder does not matter much.
            char key = input_text[i];
            bool found = false;

            for (size_t j = 0; j < encodingKeyPairs.size(); j++)
            {
                for (const auto &pair : encodingKeyPairs[j])
                {
                    if (pair.first == key)
                    {
                        found = true;
                        indices.push_back(pair.second);
                        break;
                    }
                }
                if (found)
                    break;
            }
        }

        return indices;
    };

    std::vector<char> decoder(std::vector<int> &indices)
    {
        std::vector<char> decoded;
        decoded.reserve(indices.size());
        for (size_t i = 0; i < indices.size(); i++)
        {
            bool found = false;
            char key;
            for (size_t j = 0; j < encodingKeyPairs.size(); j++)
            {
                for (const auto &pair : encodingKeyPairs[j])
                {
                    if (pair.second == indices[i])
                    {
                        found = true;
                        key = pair.first;
                        break;
                    }
                }
                if (found)
                    break;
            }

            // if should find as long as we feed it with soemthing in vocab
            // please never mind my skills in this lang this is my first time learning it.
            // I might do things in the wrong way!

            if (found)
            {
                decoded.push_back(key);
            }
        }
        return decoded;
    }

    std::string decoder(std::vector<std::unordered_map<char, int>> &encoded_pairs, int N)
    {
        return "";
    }
};

class Utility
{
public:
    Utility()
    {
    }

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

    template <typename T>
    void showVector(std::vector<T> &arr)
    {
        for (size_t i = 0; i < arr.size(); i++)
        {
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }
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
    // For now lets make this method convert the flat array which is computed by the GPU to 2D array of vectors
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

    // IMPORTANT NOTE HERE:-
    // opreation in the parallel happens through the flat strip of memory so just to check and see I am writing this.
    // This is a performace bottlneck in the code but for the sake of learning you cant really think in terms of flat memory.
    // I am keeping this here. Once the model is working we will modifiy and make this flat we might.
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

    // this is for debugging multi headed attention
    // errors in the GPU are quiet, like me
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

class EncoderText
{
    std::vector<char> fileAsChar;

public:
    void loadTextChunk(std::string &path)
    {
        std::ifstream file(path);

        if (!file.is_open())
        {
            std::runtime_error("Failed to open the file");
        }

        std::vector<char> charArray;
        char ch;

        while (file.get(ch))
        {
            charArray.push_back(ch);
        }

        file.close();
        this->fileAsChar = charArray;
    }

    const std::vector<char> &getFileAsChar() const
    {
        return fileAsChar;
    }
};

struct posDataPtr
{
    int s1;
    int s2;
} dataPointerTrack;

struct IO // this x, and y are stored for one batch.
{
    std::vector<std::vector<int>> x;
    std::vector<std::vector<int>> y;

    bool empty() const
    {
        return x.empty() && y.empty();
    }
} io;

struct Batch
{
    int *x; // shape (seq_len, batch_size)  pointers into flat memory
    int *y;
    int seq_len;
    int width;
    bool empty() const { return x == nullptr && y == nullptr; }
};

// For this transformer our goal is to learn things so we will create a simple data loader, and feed it with toy data.
// For this particular case lets user silding window to retrive the data in batch.
// Why not kernel launch for this, if this happens in 1ms then its fine, this happens only once not something that happens all the time.
class DataLoader
{
public:
    int batch_size;
    posDataPtr batchPointer;
    int *totalDataX = nullptr;
    int *totalDataY = nullptr;
    int filePointerX;
    bool drop_last;
    const std::vector<int> &data;
    int seq_len;
    std::unique_ptr<Utility> utils = std::make_unique<Utility>();
    int currentIterator;

    int *xBatch;
    int *yBatch;

    int totalIterations;

private:
    // I am not sure how I am I going to explain it to you when I am in the state of flow.
    // Even I wont understand this after a while I need to think deep.
    void populateColsInBatch(int &filePointerX, std::vector<std::vector<int>> &x, std::vector<std::vector<int>> &y) // lets make this return y.
    {
        int row = x.size();
        int cols = x[0].size();

        for (int i = 0; i < cols; ++i)
        {
            for (int j = 0; j < row; ++j)
            {
                if (!(filePointerX <= data.size() - 1)) // making sure that never reaches the end, so that we can prepare y acoordingly.
                {
                    filePointerX = 0;
                }

                x[j][i] = this->data[filePointerX];
                y[j][i] = this->data[(filePointerX + 1) <= data.size() ? (filePointerX + 1) : 0]; // loop back around this is one of the solution.
                filePointerX++;                                                                   // this approach is obviously not fissible and flexible if you are using different kind of tokenizer
            }
        }
    }

    // We can consider this like a iterator, again I am a beginner I go in flow state in my first project.
    IO getData()
    {
        int dataSize = data.size();
        int pt2Inc = batch_size;

        std::unique_ptr<IO> ioSeq = std::make_unique<IO>();

        // Check for the edge case of data being empty;
        if (data.empty())
        {
            throw std::invalid_argument("Data is empty");
        }
        if (batch_size > dataSize)
        {
            throw std::invalid_argument("We wont deal with batch_size greater than data_size case at the moment.");
        }

        if (this->batchPointer.s2 > dataSize)
        {
            return {};
        }

        int decisionHeight = dataSize - (dataSize % batch_size);

        if (decisionHeight == this->batchPointer.s2) // this will be 0 if the data goes in cleanly
        {
            // one bug was here if 891 == 891 example then without taking out the slice we are returning empty that wont work
            if (drop_last == true)
            {
                // std::cout << dataPointer.s1 << " : " << decisionHeight << std::endl;
                this->batchPointer.s2 += decisionHeight; // if not this then it will return infinitely just make this grater than data size.

                const int cols = decisionHeight - batchPointer.s1;

                // std::cout << " Roes: " << seq_len << " Cols: " << cols << std::endl;

                std::vector<std::vector<int>> vecX(this->seq_len, std::vector<int>(cols)); // I can think the hardway here but I am not sure if that's the right appoprach.
                std::vector<std::vector<int>> vecY(this->seq_len, std::vector<int>(cols));
                populateColsInBatch(filePointerX, vecX, vecY);
                io.x = vecX;
                io.y = vecY;
                return io;
            }
            else
            {
                // if the drop_last is false then we increment the pointer2 by remaining amount
                pt2Inc = (dataSize % batch_size);
                const int cols = this->batchPointer.s2 - this->batchPointer.s1;

                // std::cout << " Roes: " << seq_len << " Cols: " << cols << std::endl;

                std::vector<std::vector<int>> vecX(this->seq_len, std::vector<int>(cols)); // this is for the x
                std::vector<std::vector<int>> vecY(this->seq_len, std::vector<int>(cols)); // same size for the y
                populateColsInBatch(filePointerX, vecX, vecY);

                this->batchPointer.s1 += batch_size;
                this->batchPointer.s2 += pt2Inc;

                io.x = vecX;
                io.y = vecY;
                return io;
            }
        }

        const int cols = this->batchPointer.s2 - this->batchPointer.s1;

        // std::cout << " Roes: " << seq_len << " Cols: " << cols << std::endl;

        // std::cout << " Pointer 1: " << this->batchPointer.s2 << " Pointer 2: " << this->batchPointer.s1 << std::endl;

        std::vector<std::vector<int>> vecX(this->seq_len, std::vector<int>(cols));
        std::vector<std::vector<int>> vecY(this->seq_len, std::vector<int>(cols));
        populateColsInBatch(filePointerX, vecX, vecY);

        this->batchPointer.s1 += batch_size;
        this->batchPointer.s2 += pt2Inc;

        io.x = vecX;
        io.y = vecY;
        return io;
    }

    void getBatch()
    {

        IO currentBatch; // this thing hold a certian buffer for that IO object but we will work on returning a flat memory

        int totalIterations = (data.size() + batch_size - 1) / batch_size;

        int lastBatch = data.size() % batch_size;
        int fullBatches = data.size() / batch_size;
        int totalElements = (fullBatches * batch_size * seq_len) + (lastBatch * seq_len);

        this->totalDataX = (int *)malloc(totalElements * sizeof(int));
        this->totalDataY = (int *)malloc(totalElements * sizeof(int));
        // so we have data in batch with each Shape (seq_len, batch_size)

        int offsetX = 0;
        int offsetY = 0;

        while (!(currentBatch = getData()).empty())
        {
            // totalData Shape(seq_len, batch_size)

            // check the shape for each
            // utils->Print2DVector(currentBatch.x);
            for (const auto &row : currentBatch.x)
            { // its important to make this contiguous
                std::memcpy(this->totalDataX + offsetX, row.data(), row.size() * sizeof(int));
                offsetX += row.size();
            }

            for (const auto &row : currentBatch.y)
            {
                //                where to write           src           size
                std::memcpy(this->totalDataY + offsetY, row.data(), row.size() * sizeof(int));
                offsetY += row.size(); // this->totalDataY + offsetY write there
            }
        }
    }

public:
    DataLoader(int batch_size, const std::vector<int> &data, int seq_len, bool drop_last = true)
        : batch_size(batch_size), data(data), drop_last(drop_last)
    {
        batchPointer.s1 = 0;
        batchPointer.s2 = batch_size;
        this->seq_len = seq_len;

        xBatch = new int[seq_len * batch_size];
        yBatch = new int[seq_len * batch_size];
        this->getBatch();
        currentIterator = 0;

        totalIterations = (data.size() + batch_size - 1) / batch_size;
    }

    ~DataLoader()
    {
        (totalDataX != nullptr ? free(totalDataX) : void());
        (totalDataY != nullptr ? free(totalDataY) : void());
    }

    // I almost forgot about that siliding window, hang tight
    // previosuly I was doing batch by batch but that is costly
    // espically in parallel that goes through PCIe BUS which is slow.

    void printData(std::string input)
    {

        int lastBatch = data.size() % batch_size;
        int totalElements = ((totalIterations - 1) * batch_size * seq_len) + lastBatch * seq_len;
        int total_samples = totalElements / seq_len;

        for (int i = 0; i < total_samples; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                int token = input == "x" ? totalDataX[i * seq_len + j] : totalDataY[i * seq_len + j];
                std::cout << token << " ";
            }
            std::cout << "\n";
        }
    }

    /*
        This looks scary but the logic is pretty straightforward, even I might not get at instantly after few months,
        but it is what it is because it has 100 reason to screw up.
    */

    Batch iter()
    {
        // this is the tricky part here
        // because if drop_last = false then we might have value that does not fit cleanly
        // infact its bothering me for a while now but the logic is,

        if (currentIterator >= totalIterations)
            return {nullptr, nullptr};

        int currentWidth = (currentIterator == totalIterations - 1 && data.size() % batch_size != 0)
                               ? data.size() % batch_size
                               : batch_size; // if this is in the last iteration
        if (drop_last && currentWidth < batch_size)
        {
            return {nullptr, nullptr}; // signals done
        }

        int fullBatches = data.size() / batch_size;

        int offset = (currentWidth == batch_size)
                         ? currentIterator * seq_len * batch_size // full batch
                         : fullBatches * seq_len * batch_size;

        // fill xBatch
        for (int row = 0; row < seq_len; ++row)
        {
            for (int col = 0; col < currentWidth; ++col)
            {
                int idx = offset + (row * currentWidth) + col;

                xBatch[row * currentWidth + col] = totalDataX[idx];
                yBatch[row * currentWidth + col] = totalDataY[idx];
            }
        }

        // std::cout << "Iteration count: " << currentIterator
        // << " batch width: " << currentWidth
        // << std::endl;

        currentIterator++;
        return {xBatch, yBatch, seq_len, currentWidth}; // return actual width too
    }

    void resetIterator()
    {
        this->currentIterator = 0;
    }
};

// in the phase of learning I did it myway shouldn't be like this,
// make it contangeous even though not initllized in the GPU
class Initializer
{
public:
    std::vector<std::vector<float>> HeInit(int vocabSize, int dModel)
    {
        float stdDev = std::sqrt(2.0f / dModel);

        std::vector<std::vector<float>> tokenEmbeddings(
            vocabSize,
            std::vector<float>(dModel));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> gaussian(0.0f, 1.0f); // mean 0 std dev

        for (int i = 0; i < vocabSize; ++i)
        {
            for (int j = 0; j < dModel; ++j)
            {
                tokenEmbeddings[i][j] = gaussian(gen);
            }
        }
        return tokenEmbeddings;
    }
};