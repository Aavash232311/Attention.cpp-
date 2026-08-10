#pragma once
#include <iostream>



using namespace std;

// in the phase of learning I did it myway shouldn't be like this,
// make it contangeous even though not initllized in the GPU
class Initializer
{
public:
    vector<std::vector<float>> HeInit(int vocabSize, int dModel)
    {
        float stdDev = std::sqrt(2.0f / dModel);

        vector<vector<float>> tokenEmbeddings(
            vocabSize,
            vector<float>(dModel));

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> gaussian(0.0f, 1.0f); // mean 0 std dev

        for (int i = 0; i < vocabSize; ++i)
        {
            for (int j = 0; j < dModel; ++j)
            {
                tokenEmbeddings[i][j] = gaussian(gen);
            }
        }
        return tokenEmbeddings;
    }
};