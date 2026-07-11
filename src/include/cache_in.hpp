#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using namespace std;

using json = nlohmann::json;
/*
    First debugger call from python will write the hp in json format,
    instead of passing in the paramaters here which will make it look not so clean
    we can directly load the josn here
*/

struct Hyperparamaters
{
    int d_model;
    int vocab_size;
    int seq_len;
    int num_heads;
    int batch_size;
};


// why would we want to do something like this?
// to load hyperparamaters in the seed that we are trying to initlize

Hyperparamaters readHyperparameters()
{
    std::string path = "./src/cache/config.json";
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cerr << "Failed to open: " << path << '\n';
        throw exception();
    }

    json data;
    file >> data;

    Hyperparamaters paramaters = {
        data["d_model"],
        data["vocab_size"],
        data["seq_len"],
        data["seq_len"],
        data["batch_size"]
    };

    return paramaters;
}