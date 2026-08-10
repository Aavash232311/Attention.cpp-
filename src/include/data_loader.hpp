#pragma once
#include <iostream>
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


inline struct posDataPtr
{
    int s1;
    int s2;
} dataPointerTrack;



inline struct IO // this x, and y are stored for one batch.
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

