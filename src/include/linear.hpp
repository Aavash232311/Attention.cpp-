#pragma once
struct LinearParams // just binary pointer can be used for layer norm or anything that contains two paramater pairs
{
    float *Weight;
    float *Bias;
};