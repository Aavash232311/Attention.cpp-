#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <unordered_map>

class Helper
{

public:
    std::vector<std::unordered_map<char, int>> encoder(std::string input_text)
    {
        int input_lengh = input_text.length();

        std::vector<std::unordered_map<char, int>> encoded_pair(input_lengh);

        for (size_t i = 0; i < input_lengh; i++)
        {
            encoded_pair[i][input_text[i]] = i;
        }
        return encoded_pair;
    };

    void showHashMap(std::vector<std::unordered_map<char, int>> &encoded_input, int N)
    {
        for (size_t i = 0; i < N; i++)
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