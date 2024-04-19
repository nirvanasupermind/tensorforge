#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <random>
#include <sstream>
#include <algorithm>
#include "tensor.h"

namespace tfg {
    template <typename T = double>
    class Distribution {
    public:
        virtual Tensor<T> cdf(const Tensor<T>& val) const;
        virtual Tensor<T> sample(const std::vector<size_t> &shape) const;
    };

    template <typename T = double>
    class Bernoulli {
    public:
        Tensor<T> probs;
        Bernoulli(const Tensor<T> &probs)
            : probs(probs) {    
        }
        Tensor<T> cdf(const Tensor<T>& val) const {
            return val.binary_transform(probs, [](T a, T b) {
                return ((a < 0.0) ? 0.0 : ((a < 1.0) ? (1.0 - b) : 1.0));
            });
        }
        Tensor<T> sample(const std::vector<size_t> &shape) const {
            return probs.transform([shape](T a) {
                T rand_num = Tensor<T>::random(shape).item();
                return rand_num < a ? 0.0 : 1.0;
            });
        }
    };
}

#endif