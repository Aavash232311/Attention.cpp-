#pragma once

class Autograd
{
public:
    Autograd();

    void backward();
    void zero_grad();
};