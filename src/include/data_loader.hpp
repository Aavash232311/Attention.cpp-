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

struct Batch
{
    int *x; // shape (seq_len, batch_size)  pointers into flat memory
    int *y;

    int height;
    int width;

    bool empty() const { return x == nullptr && y == nullptr; }
};

enum Device
{
    GPU,
    CPU,
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
    Device device = GPU;

    void loadBatches()
    {
        int total_samples = static_cast<int>(data.size()) - seq_len;
        int totalIterations = total_samples / batch_size;

        int iterator = 0;

        for (int i = 0; i < totalIterations; ++i)
        {
            std::cout << "\nBatch " << i << '\n';

            for (int j = 0; j < batch_size; ++j)
            { // no drop last here, I will take care of that one later.
                vector<int> current_x(
                    data.begin() + iterator,
                    data.begin() + iterator + seq_len);

                vector<int> current_y(
                    data.begin() + iterator + 1,
                    data.begin() + iterator + seq_len + 1);

                    

                iterator++;
            }
        }
    }

public:
    DataLoader(int batch_size, int seq_len, const std::vector<int> &data)
        : batch_size(batch_size), seq_len(seq_len), data(data)
    {
        loadBatches();
    }

    void to(string device)
    {
        if (device == "cpu")
        {
            this->device = CPU;
        }
        else if (device == "gpu")
        {
            this->device = GPU;
        }
        else
        {
            std::runtime_error("Load on either GPU/CPU");
        }
    }
};
