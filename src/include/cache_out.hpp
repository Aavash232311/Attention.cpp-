#include <memory>
#include <cstdio>
#include <tuple>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem;

template <typename T>
void releaseFile(
    const string &filename,
    const T *data,
    size_t num_elements)
{
    const string path = "./src/cache/cpp_out/" + filename;

    ofstream out(path, ios::binary);

    out.write(reinterpret_cast<const char *>(data), num_elements * sizeof(T));
}

string jsonEscape(const string &s)
{
    string out;
    out.reserve(s.size());
    for (char c : s)
    {
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\r':
            out += "\\r";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

void releaseConfig(
    const vector<string> key,
    const vector<int> value)
{
    const string path = "./src/cache/config.json";

    std::ofstream out(path);
    if (!out.is_open())
    {
        throw std::runtime_error("releaseConfig: failed to open " + path);
    }

    out << "{\n";
    for (size_t i = 0; i < key.size(); ++i)
    {
        out << "  \"" << jsonEscape(key[i]) << "\": " << value[i];
        if (i != key.size() - 1)
            out << ",";
        out << "\n";
    }
    out << "}\n";

    out.close();
}

// Realeses json for hyperparamaters config

void releaseHyperParamaters(
    int d_model,
    int vocab_size,
    int batch_size,
    int seq_len,
    int num_heads
)
{
    std::vector<string> key = {
        "d_model",
        "vocab_size",
        "batch_size",
        "seq_len",
        "num_heads"};

    std::vector<int> value = {
        d_model,
        vocab_size,
        batch_size,
        seq_len,
        num_heads};

    releaseConfig(key, value);
}


// For debugging the autograd engine we would need to release entire

// only array from RAM
template <typename T>
void bulkRelease(const vector<tuple<T*, int, string>>& items) {
    for (const auto& [arr, size, filename] : items) {
        releaseFile<T>(filename, arr, size);
    }
}
