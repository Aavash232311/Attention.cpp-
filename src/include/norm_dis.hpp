#pragma once
#include <iostream>

using namespace std;


// make it contangeous even though not initllized in the GPU
class Initializer
{
public:
    vector<vector<float>> EmbeddingInit(int vocabSize, int dModel)
    {
        std::mt19937 gen(1234);
        float stdDev = 0.9f; // GPT style; use std::sqrt(2.0f / dModel) for He init

        normal_distribution<float> gaussian(0.0f, stdDev);

        vector<vector<float>> tokenEmbeddings(vocabSize, vector<float>(dModel));

        for (int i = 0; i < vocabSize; ++i)
            for (int j = 0; j < dModel; ++j)
                tokenEmbeddings[i][j] = gaussian(gen);

        return tokenEmbeddings;
    }
};