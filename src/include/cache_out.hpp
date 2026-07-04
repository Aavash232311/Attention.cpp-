#include <memory>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;


void releaseFile(
    const std::string& filename,
    const float* data,
    size_t num_elements
)
{
    const string path = "./src/cache/" + filename;

    std::ofstream out(path, std::ios::binary);

    out.write(reinterpret_cast<const char*>(data), num_elements * sizeof(float));
}