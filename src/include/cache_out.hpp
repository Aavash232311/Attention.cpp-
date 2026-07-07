#include <memory>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

void releaseFile(
    const std::string &filename,
    const float *data,
    size_t num_elements)
{
    const string path = "./src/cache/" + filename;

    std::ofstream out(path, std::ios::binary);

    out.write(reinterpret_cast<const char *>(data), num_elements * sizeof(float));
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