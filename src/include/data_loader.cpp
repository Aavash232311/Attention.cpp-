//
// Created by avash on 11/18/26.
//
#include "data_loader.hpp"


void EncoderText::loadTextChunk(std::string &path) {
    std::ifstream file(path);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open the file");
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

void DataLoader::loadBatches() {
    int total_samples = std::max(0, static_cast<int>(data.size()) - seq_len);
    int totalIterations = total_samples / batch_size;

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

            // Generation slides its context by one token. Train on those
            // same offsets instead of only every seq_len-th window.
            ++iterator;
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

DataLoader::DataLoader(int batch_size, int seq_len, const std::vector<int> &data, Device device) :
batch_size(batch_size), seq_len(seq_len), data(data), device(device) {
    loadBatches();
}


void DataLoader::inspectData() {
    for (size_t i = 0; i < batches.size(); ++i)
    {
        utils->printFlatArray2D(batches[i].x, batch_size, seq_len, "device");
    }
}

DataLoader::~DataLoader() {

    for (size_t i = 0; i < batches.size(); ++i)
    {
        int *x = batches[i].x;
        int *y = batches[i].y;

        cudaFree(x);
        cudaFree(y);
    }
}