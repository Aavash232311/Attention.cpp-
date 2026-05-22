#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_map>

class Helper
{
private:
    std::vector<std::unordered_map<char, int>> encodingKeyPairs;

public:
    Helper(std::vector<char> &encodingPool)
    {
        // okay so we have this array of characters based on their poistion first let's encode them.
        encodingKeyPairs.resize(encodingPool.size()); // resize

        for (size_t i = 0; i < encodingPool.size(); i++)
        {
            encodingKeyPairs[i][encodingPool[i]] = i;
        }
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

    void print_full_matrix(const float *matrix, int seq_len, int d_model)
    {

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "[\n";
        for (int r = 0; r < seq_len; ++r)
        {
            std::cout << "  [";
            for (int c = 0; c < d_model; ++c)
            {

                std::cout << std::setw(8) << matrix[r * d_model + c];
                if (c < d_model - 1)
                {
                    std::cout << ", ";
                }
            }
            std::cout << "]";

            if (r < seq_len - 1)
            {
                std::cout << ",\n";
            }
        }
        std::cout << "\n] Shape: (" << seq_len << ", " << d_model << ")\n";
    }

    template <typename T>
    void showVector(std::vector<T> &arr)
    {
        for (size_t i = 0; i < arr.size(); i++)
        {
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }

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

    std::vector<char> loadTextChunk(std::string &path)
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

struct posDataPtr
{
    int s1;
    int s2;
} dataPointerTrack;

// For this transformer our goal is to learn things so we will create a simple data loader, and feed it with toy data.
// For this particular case lets user silding window to retrive the data in batch.
class DataLoader
{
    std::string data_path;
    int batch_size;
    posDataPtr dataPointer;
    bool drop_last;
    std::vector<char> data;

public:
    DataLoader(std::string data_path, int batch_size, bool drop_last = false)
    {
        // pass by value, it does not matter for such small value
        this->batch_size = batch_size;
        this->data_path = data_path;

        this->dataPointer.s2 += batch_size;
    }
    // We can consider this like a iterator
    int getData()
    {
        // Check for the edge case of data being empty;
        if (data.empty())
        {
            std::cout << " The data is empty. " << std::endl;
            return 1;
        }

        // If we want to send all the batches inside then the batch_size should divide the dataset cleanly
        bool fullBatch = false;
        if (batch_size % data.size() == 0)
            fullBatch = true;

        if (fullBatch == true)
        {
            this->dataPointer.s1 += batch_size;
            this->dataPointer.s2 += batch_size;
        }
        else // This condition is triggrred only when fullBatch is false.
        {
            // If it does not divide cleanly then we need to watchout for the dropout.
            int projectSecondPointer;
            // The first pointer is going to be as it is.
            this->dataPointer.s1 += batch_size;
            if (this->drop_last == false)
            {
                projectSecondPointer = this->dataPointer.s2 + this->batch_size;

                if (projectSecondPointer > data.size())
                {
                    // This is the edge case where we have that uneven fit.
                    this->dataPointer.s2 = this->data.size() - this->batch_size;
                }
            }
            else
            {
                // If we have drop last is true, then we ignore that chunk of data
                projectSecondPointer = this->dataPointer.s2 + this->batch_size;
            }
            this->dataPointer.s2 += projectSecondPointer;
        }
        
        return 0;
    }
};