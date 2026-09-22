#pragma once
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./optimizer_mem.hpp"

struct AdamWMemConfig
{
public:
    AdamWPhysics dl_dw;

    AdamWMemConfig()
    {
        dl_dw.init(1000);
    }
};