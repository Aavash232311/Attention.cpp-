#pragma once
#include <math.h>
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

using namespace std;

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

enum Device
{
    GPU,
    CPU,
};

struct Batch
{
    int *x; // shape (seq_len, batch_size)  pointers into flat memory
    int *y;

    int height;
    int width;

    bool empty() const { return x == nullptr && y == nullptr; }
};

// For this transformer our goal is to learn things so we will create a simple data loader, and feed it with toy data.
// For this particular case lets user silding window to retrive the data in batch.
// Why not kernel launch for this, if this happens in 1ms then its fine, this happens only once not something that happens all the time.
class DataLoader
{
private:
    unique_ptr<Utility> utils = make_unique<Utility>();
    int batch_size;
    int seq_len;

    const vector<int> &data;
    Device device = CPU;
    vector<Batch> batches;

    void loadBatches()
    {

        int total_samples = static_cast<int>(data.size()) - seq_len;
        int totalIterations = total_samples / (batch_size * seq_len);

        int iterator = 0;

        batches.reserve(totalIterations);

        for (int i = 0; i < totalIterations; ++i)
        {
            // shape (batch_size, seq_len)

            vector<int> id_x(batch_size * seq_len);
            vector<int> id_y(batch_size * seq_len);

            for (int j = 0; j < batch_size; ++j)
            { // no drop last here, I will take care of that one later.
                vector<int> current_x(
                    data.begin() + iterator,
                    data.begin() + iterator + seq_len);

                vector<int> current_y(
                    data.begin() + iterator + 1,
                    data.begin() + iterator + seq_len + 1);

                for (int k = 0; k < current_x.size(); ++k)
                {
                    int idx = (seq_len * j) + k;

                    id_x[idx] = current_x[k];
                    id_y[idx] = current_y[k];
                }

                iterator += seq_len; // prevents overlap
            }

            int *strip_x = utils->vectorToFlatStrip(id_x, "gpu");
            int *strip_y = utils->vectorToFlatStrip(id_y, "gpu");

            int *b_x;
            int *b_y;

            cudaMalloc((void **)&b_x, batch_size * seq_len * sizeof(int));
            cudaMalloc((void **)&b_y, batch_size * seq_len * sizeof(int));

            cudaMemcpy(b_x, strip_x, batch_size * seq_len * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(b_y, strip_y, batch_size * seq_len * sizeof(int), cudaMemcpyDeviceToDevice);

            batches.push_back(Batch{b_x, b_y, batch_size, seq_len});

            cudaFree(strip_x);
            cudaFree(strip_y);
        }
    }

public:
    DataLoader(int batch_size, int seq_len, const std::vector<int> &data, Device device = GPU)
        : batch_size(batch_size), seq_len(seq_len), data(data), device(device)
    {

        loadBatches();
    }

    vector<Batch> &getBatches()
    {
        return batches;
    }

    void inspectData()
    {
        int total_samples = static_cast<int>(data.size()) - seq_len;
        int totalIterations = total_samples / (batch_size * seq_len);

        for (int i = 0; i < totalIterations; ++i)
        {
            utils->printFlatArray2D(batches[i].x, batch_size, seq_len, "device");
        }
    }

    ~DataLoader()
    {
        int total_samples = static_cast<int>(data.size()) - seq_len;
        int totalIterations = total_samples / (batch_size * seq_len);

        for (int i = 0; i < totalIterations; ++i)
        {
            int *x = batches[i].x;
            int *y = batches[i].y;

            cudaFree(x);
            cudaFree(y);
        }
    }
};
