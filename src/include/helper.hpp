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
#include <stdexcept>
#include <iosfwd>

using namespace std;

class Helper
{
private:
    std::vector<char> vocab;

    bool searchVocab(char c);

    void populateVocab(std::vector<char> &encodingPool);

    int characterKeyLookup(char c);

public:
    Helper(std::vector<char> encodingPool);
    // getter to get the encoding pool of everything that we plan to encode on.
    std::vector<int> getEncodedList();

    std::vector<int> encoder(std::string &input_text);
    // this works fine.
    std::vector<char> decoder(std::vector<int> &indices);
    std::vector<std::unordered_map<char, int>> encodingKeyPairs;

    void inspect();

    char decodePosition(int index);

    int vocab_size();

    friend ostream &operator<<(ostream &, const Helper &helper);
};
