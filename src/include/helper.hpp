#pragma once
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

// I will be really honest it takes time for me to think deeply, but I will figure it out
// till this day AI hallucination is common on task that requires deep thinking.
// whatever even if this market is brutal I am investing my effort into thinking.
class Helper
{
private:
    std::vector<std::unordered_map<char, int>> encodingKeyPairs;

public:
    Helper(std::vector<char> encodingPool)
    {
        // okay so we have this array of characters based on their poistion first let's encode them.
        encodingKeyPairs.resize(encodingPool.size()); // resize

        for (size_t i = 0; i < encodingPool.size(); i++)
        {
            encodingKeyPairs[i][encodingPool[i]] = i;
        }
    }
    // getter to get the encoding pool of everything that we plan to encode on.
    std::vector<int> getEncodedList()
    {
        std::vector<int> newList;
        newList.reserve(encodingKeyPairs.size());

        for (size_t i = 0; i < encodingKeyPairs.size(); i++)
        {
            for (const auto &pair : encodingKeyPairs[i])
            {
                newList.push_back(pair.second);
            }
        }

        return newList;
    }

    std::vector<int> encoder(std::string &input_text)
    {
        // let's iterate on what we have and we will check for the encoded dict.
        std::vector<int> indices;
        indices.reserve(input_text.length());

        for (size_t i = 0; i < input_text.length(); i++)
        {
            // let's just do a linear search here. It's just a encoder/decoder does not matter much.
            char key = input_text[i];
            bool found = false;

            for (size_t j = 0; j < encodingKeyPairs.size(); j++)
            {
                for (const auto &pair : encodingKeyPairs[j])
                {
                    if (pair.first == key)
                    {
                        found = true;
                        indices.push_back(pair.second);
                        break;
                    }
                }
                if (found)
                    break;
            }
        }

        return indices;
    };

    std::vector<char> decoder(std::vector<int> &indices)
    {
        std::vector<char> decoded;
        decoded.reserve(indices.size());
        for (size_t i = 0; i < indices.size(); i++)
        {
            bool found = false;
            char key;
            for (size_t j = 0; j < encodingKeyPairs.size(); j++)
            {
                for (const auto &pair : encodingKeyPairs[j])
                {
                    if (pair.second == indices[i])
                    {
                        found = true;
                        key = pair.first;
                        break;
                    }
                }
                if (found)
                    break;
            }

            // if should find as long as we feed it with soemthing in vocab
            // please never mind my skills in this lang this is my first time learning it.
            // I might do things in the wrong way!

            if (found)
            {
                decoded.push_back(key);
            }
        }
        return decoded;
    }

    std::string decoder(std::vector<std::unordered_map<char, int>> &encoded_pairs, int N)
    {
        return "";
    }
};
