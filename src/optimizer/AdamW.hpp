#pragma once
#include "../include/helper.hpp"
#include "../include/utils.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>



using namespace std;


class AdamW {
    public:
        bool debug = false;

        AdamW(
            bool debug
        )
        {
            this->debug = debug;
        }

    void step()
    {
        if (debug)
        {
            cout << "Invoked AdamW optimizer" << endl;
        }
    }
};