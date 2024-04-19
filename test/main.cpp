#include <iostream>
#include <vector>
#include <complex>
#include "../src/tensor.h"

int main() {
    tfg::Tensor<> x({{1,2},{3,4},{5,6}});
    std::cout << x.subtensor({0,0},{2,1}).str() << '\n';
}