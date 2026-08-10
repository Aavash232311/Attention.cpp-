#pragma once

#include <vector>
#include <fstream>
#include <stdexcept>

using namespace std;

inline float* load_bin(const std::string& path, size_t num_elements) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    float* data = new float[num_elements];  // stream reads in bytes char = 1 byte
    file.read(reinterpret_cast<char*>(data), num_elements * sizeof(float));  

    if (file.gcount() != static_cast<std::streamsize>(num_elements * sizeof(float))) {
        delete[] data;
        throw std::runtime_error("File size mismatch for: " + path);
    }

    return data;
}