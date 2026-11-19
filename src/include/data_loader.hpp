#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include "utils.hpp"
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include <fstream>


using namespace std;

class EncoderText
{
    std::vector<char> fileAsChar;

public:
    void loadTextChunk(std::string &path);
    [[nodiscard]] const std::vector<char> &getFileAsChar() const {return fileAsChar;}

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
// For this particular case lets user sliding window to retrieve the data in batch.
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

    void loadBatches();

public:
    DataLoader(int batch_size, int seq_len, const std::vector<int> &data, Device device = GPU);

    vector<Batch> &getBatches(){ return batches;}

    void inspectData();

    ~DataLoader();
};
