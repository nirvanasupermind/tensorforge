#include <iostream>
#include <vector>
#include <complex>
#include "../src/tensor.h"
#include "../src/distributions.h"

int main() {
    tfg::Bernoulli<> d(tfg::Tensor<>({0.5,0.5}));
    std::cout << d.sample({1}).str() << '\n';

}