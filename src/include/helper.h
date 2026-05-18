#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <unordered_map>

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

    std::vector<std::unordered_map<char, int>> encoder(std::string input_text)
    {
        // let's iterate on what we have and we will check for the encoded dict.

        auto encodedVectors = std::vector<std::unordered_map<char, int>>(input_text.length());

        for (size_t i = 0; i < input_text.length(); i++)
        {
            // let's just do a linear search here. It's just a encoder/decoder does not matter much.
            char key = input_text[i];
            bool found = false;
            int value;

            for (size_t j = 0; j < encodingKeyPairs.size(); j++)
            {
                for (const auto &pair : encodingKeyPairs[j])
                {
                    if (pair.first == key)
                    {
                        found = true;
                        value = pair.second;
                        break;
                    }
                }
                if (found)
                    break;
            }
            // I mean as along as we send string to incode that lies withing the vocab of the text it should find it.
            if (found)
            {
                encodedVectors[i][key] = value;
            }
        }
        return encodedVectors;
    };

    void showHashMap(std::vector<std::unordered_map<char, int>> &encoded_input)
    {
        for (size_t i = 0; i < encoded_input.size(); i++)
        {
            for (const auto &pair : encoded_input[i])
            {
                std::cout << "  Key: " << pair.first << " -> Value: " << pair.second << std::endl;
            }
        }
        return;
    }

    std::string decoder(std::vector<std::unordered_map<char, int>> &encoded_pairs, int N)
    {
        return "";
    }
};

class EncoderText
{
public:
    EncoderText()
    {
    }

    std::vector<char> loadTextChunk(std::string path)
    {
        std::ifstream file(path);

        if (!file.is_open())
        {
            std::runtime_error("Failed to open the file");
        }

        std::vector<char> charArray;
        char ch;

        while (file.get(ch))
        {
            charArray.push_back(ch);
        }

        file.close();
        return charArray;
    }
};