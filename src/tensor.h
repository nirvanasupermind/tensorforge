#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <random>
#include <sstream>
#include <algorithm>

namespace tpp {
    class TPPException : public std::exception {
    public:
        std::string msg;
        TPPException(const std::string& msg)
            : msg(msg) {
        }
        char* what() {
            return (char*)(msg.c_str());
        }
    };

    template <typename T = double>
    class Tensor {
    public:
        // Member
        std::vector<size_t> shape{};
        std::vector<T> data{};

        // Constructors
        Tensor(const std::vector<size_t>& shape, const std::vector<T>& data)
            : shape(shape), data(data) {
        }

        Tensor(const std::vector<T>& vec)
            : shape(std::vector<size_t>{vec.size()}), data(vec) {
        }

        Tensor(const std::vector<std::vector<T> >& mat)
            : shape(std::vector<size_t>{mat.size(), mat.at(0).size()}) {
            for (int i = 0; i < mat.size(); i++) {
                for (int j = 0; j < mat.at(0).size(); j++) {
                    data.push_back(mat.at(i).at(j));
                }
            }
        }

        Tensor(const std::initializer_list<std::initializer_list<T> >& mat) {
            std::vector<std::initializer_list<T> > mat_vec(mat);
            shape = { mat_vec.size(), std::vector<T>(mat_vec.at(0)).size() };
            for (int i = 0; i < shape.at(0); i++) {
                std::vector<T> row_vec(mat_vec.at(i));
                for (int j = 0; j < shape.at(1); j++) {
                    data.push_back(row_vec.at(j));
                }
            }
        }

        Tensor(const std::vector<std::vector<std::vector<T> > >& cube)
            : shape(std::vector<size_t>{cube.size(), cube.at(0).size(),
                cube.at(0).at(0).size()}) {
            for (int i = 0; i < cube.size(); i++) {
                for (int j = 0; j < cube.at(0).size(); j++) {
                    for (int k = 0; k < cube.at(0).at(0).size(); k++) {
                        data.push_back(cube.at(i).at(j).at(k));
                    }
                }
            }
        }

        Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<T> > >& cube) {
            std::vector<std::initializer_list<std::initializer_list<T> > > cube_vec(cube);
            std::vector<std::initializer_list<T> > first_mat_vec(cube_vec.at(0));
            shape = { cube_vec.size(), first_mat_vec.size(), std::vector<T>(first_mat_vec.at(0)).size() };
            for (int i = 0; i < cube.size(); i++) {
                std::vector<std::initializer_list<T> > mat_vec(cube_vec.at(i));
                for (int j = 0; j < mat_vec.size(); j++) {
                    std::vector<T> row_vec(mat_vec.at(j));
                    for (int k = 0; k < row_vec.size(); k++) {
                        data.push_back(row_vec.at(k));
                    }
                }
            }
        }


        // Elementwise functions
        Tensor<T> operator-() const {
            return transform([](T a) {
                return -a;
                });
        }

        Tensor<T> operator+(const Tensor& other) const {
            return binary_transform(other, [](T a, T b) {
                return a + b;
                });
        }

        Tensor<T> operator+(T other) const {
            return binary_transform(other, [](T a, T b) {
                return a + b;
                });
        }

        Tensor<T> operator-(const Tensor& other) const {
            return binary_transform(other, [](T a, T b) {
                return a - b;
                });
        }

        Tensor<T> operator-(T other) const {
            return binary_transform(other, [](T a, T b) {
                return a - b;
                });
        }

        Tensor<T> operator*(const Tensor& other) const {
            return binary_transform(other, [](T a, T b) {
                return a * b;
                });
        }

        Tensor<T> operator*(T other) const {
            return binary_transform(other, [](T a, T b) {
                return a * b;
                });
        }

        Tensor<T> operator/(const Tensor& other) const {
            return binary_transform(other, [](T a, T b) {
                return a / b;
                });
        }

        Tensor<T> operator/(T other) const {
            return binary_transform(other, [](T a, T b) {
                return a / b;
                });
        }

        Tensor<T> sin() const {
            return transform([](T a) {
                return std::sin(a);
                });
        }

        Tensor<T> cos() const {
            return transform([](T a) {
                return std::cos(a);
                });
        }


        Tensor<T> tan() const {
            return transform([](T a) {
                return std::tan(a);
                });
        }

        Tensor<T> arcsin() const {
            return transform([](T a) {
                return std::asin(a);
                });
        }


        Tensor<T> arccos() const {
            return transform([](T a) {
                return std::acos(a);
                });
        }


        Tensor<T> arctan() const {
            return transform([](T a) {
                return std::atan(a);
                });
        }

        Tensor<T> arctan2(const Tensor& other) const {
            return binary_transform(other, [](T a, T b) {
                return std::atan2(a, b);
                });
        }

        Tensor<T> arctan2(T other) const {
            return binary_transform(other, [](T a, T b) {
                return std::atan2(a, b);
                });
        }

        Tensor<T> sinh() const {
            return transform([](T a) {
                return std::sinh(a);
                });
        }

        Tensor<T> cosh() const {
            return transform([](T a) {
                return std::cosh(a);
                });
        }

        Tensor<T> tanh() const {
            return transform([](T a) {
                return std::tanh(a);
                });
        }

        Tensor<T> round() const {
            return transform([](T a) {
                return std::round(a);
                });
        }

        Tensor<T> floor() const {
            return transform([](T a) {
                return std::floor(a);
                });
        }


        Tensor<T> ceil() const {
            return transform([](T a) {
                return std::ceil(a);
                });
        }

        Tensor<T> trunc() const {
            return transform([](T a) {
                return std::trunc(a);
                });
        }


        Tensor<T> exp() const {
            return transform([](T a) {
                return std::exp(a);
                });
        }


        Tensor<T> expm1() const {
            return transform([](T a) {
                return std::expm1(a);
                });
        }


        Tensor<T> exp2() const {
            return transform([](T a) {
                return std::exp2(a);
                });
        }


        Tensor<T> log() const {
            return transform([](T a) {
                return std::log(a);
                });
        }


        Tensor<T> log10() const {
            return transform([](T a) {
                return std::log10(a);
                });
        }


        Tensor<T> log2() const {
            return transform([](T a) {
                return std::log2(a);
                });
        }

        Tensor<T> logn(T n) const {
            return transform([n](T a) {
                return std::log(a) / std::log(n);
                });
        }

        Tensor<T> log1p() const {
            return transform([](T a) {
                return std::log1p(a);
                });
        }

        Tensor<T> logaddexp(const Tensor<T>& other) const {
            return binary_transform(other, [](T a, T b) {
                return std::log(std::exp(a) + std::exp(b));
                });
        }

        Tensor<T> logaddexp(T other) const {
            return binary_transform(other, [](T a, T b) {
                return std::log(std::exp(a) + std::exp(b));
                });
        }

        Tensor<T> reciprocal() const {
            return transform([](T a) {
                return 1 / a;
                });
        }

        Tensor<T> power(const Tensor& other) const {
            return binary_transform(other, [](T a, T b) {
                return std::pow(a, b);
                });
        }

        Tensor<T> power(T other) const {
            return binary_transform(other, [](T a, T b) {
                return std::pow(a, b);
                });
        }


        Tensor<T> square() const {
            return transform([](T a) {
                return a * a;
                });
        }

        Tensor<T> sqrt() const {
            return transform([](T a) {
                return std::sqrt(a);
                });
        }

        Tensor<T> cbrt() const {
            return transform([](T a) {
                return std::cbrt(a);
                });
        }

        Tensor<bool> operator==(const Tensor& other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a == b;
                });
        }

        Tensor<bool> operator==(T other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a == b;
                });
        }

        Tensor<bool> operator!=(const Tensor& other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a != b;
                });
        }

        Tensor<bool> operator!=(T other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a != b;
                });
        }

        Tensor<bool> operator<(const Tensor& other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a < b;
                });
        }

        Tensor<bool> operator<(T other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a < b;
                });
        }

        Tensor<bool> operator<=(const Tensor& other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a <= b;
                });
        }

        Tensor<bool> operator<=(T other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a <= b;
                });
        }

        Tensor<bool> operator>(const Tensor& other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a > b;
                });
        }

        Tensor<bool> operator>=(const Tensor& other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a >= b;
                });
        }

        Tensor<bool> operator>=(T other) const {
            return bool_binary_transform(other, [](T a, T b) {
                return a >= b;
                });
        }

        // Linear algebra functions
        T dot(const Tensor& other) const {
            return operator*(other).sum();
        }

        Tensor<T> matmul(const Tensor<T>& other) const {
            std::vector<size_t> result_shape{ shape.at(0), other.shape.at(1) };
            std::vector<T> result_data;
            for (size_t i = 0; i < shape.at(0); i++) {
                for (size_t j = 0; j < other.shape.at(1); j++) {
                    T elem = 0;
                    for (size_t k = 0; k < shape.at(1); k++) {
                        elem += data.at(i * shape.at(1) + k) * other.data.at(k * other.shape.at(1) + j);
                    }
                    result_data.push_back(elem);
                }
            }
            return Tensor<T>(result_shape, result_data);
            // for (size_t i = 0; i < shape.at(0); i++) {
            //     Tensor<T> row = at({ i });
            //     for (size_t j = 0; j < other.shape.at(1); j++) {
            //         Tensor<T> col = Tensor<T>::zeros({ other.shape.at(0) });
            //         for (size_t k = 0; k < other.shape.at(0); k++) {
            //             col.data[k] = other.at({ k, j }).item();
            //         }
            //         result_data.push_back(row.dot(col));
            //     }
            // }
        }

        Tensor<T> kron(const Tensor<T>& other) {
            std::vector<size_t> result_shape{};
            std::vector<size_t> indices;
            for (size_t i = 0; i < ndim(); i++) {
                indices.push_back(i);
            }
            std::transform(indices.begin(), indices.end(), std::back_inserter(result_shape), [&](size_t i) {
                return shape.at(i) * other.shape.at(i);
                });
            std::vector<T> result_data;
            for (size_t i = 0; i < data.size(); i++) {
                for (size_t j = 0; j < other.data.size(); j++) {
                    result_data.push_back(data.at(i) * other.data.at(j));
                }
            }
            return Tensor<T>(result_shape, result_data);
        }

        Tensor<T> matrix_power(int n) const {
            if (n < 0) {
                return pinv().matrix_power(-n);
            }
            else if (n == -1) {
                return pinv();
            } if (n == 0) {
                return eye(shape.at(0));
            }
            else if (n == 1) {
                return copy();
            }
            else {
                return matmul(matrix_power(n - 1));
            }
        }

        Tensor<T> pinv(size_t num_iters = 30) const {
            Tensor<T> dbl_identity = eye(shape.at(0)) * 2.0;
            Tensor<T> xn = t().operator*(0.00001);
            for (size_t i = 0; i < num_iters; i++) {
                // std::cout << (dbl_identity - matmul(xn)).str() << '\n';
                xn = xn.matmul(dbl_identity - matmul(xn));
            }
            return xn;
        }


        Tensor<T> t() const {
            if (ndim() < 2) {
                return *this;
            }
            else {
                std::vector<size_t> result_shape{ shape.at(1), shape.at(0) };
                std::vector<T> result_data;
                for (size_t i = 0; i < shape.at(1); i++) {
                    for (size_t j = 0; j < shape.at(0); j++) {
                        result_data.push_back(data.at(j * shape.at(1) + i));
                    }
                }

                return Tensor<T>(result_shape, result_data);
            }
            // std::vector<size_t> shape_weights;
            // for (size_t i = 0; i < shape.size(); i++) {
            //     std::vector<size_t> shape_subset = { shape.end() - i, shape.end() };
            //     shape_weights.push_back(get_prod(shape_subset));
            // }

            // std::vector<size_t> result_shape(shape.rbegin(), shape.rend());
            // std::vector<T> result_data;

            // for (size_t i = 0; i < data.size(); i++) {
            //     std::vector<size_t> idx_vec;
            //     size_t temp = i;
            //     for (size_t j = 0; j < shape_weights.size(); j++) {
            //         temp /= shape_weights.at(shape_weights.size() - j - 1);
            //         std::cout << i << ' ' << shape_weights.at(shape_weights.at(shape_weights.size() - j - 1)) << ' ' << temp << '\n';
            //         idx_vec.push_back(temp);
            //     }
            //     std::vector<size_t> reversed_idx_vec(idx_vec.rbegin(), idx_vec.rend());
            //     result_data.push_back(at(reversed_idx_vec).item());
            // }

            // return Tensor<T>(result_shape, result_data);
        }

        Tensor<T> diag(size_t k = 0) const {
            if (ndim() == 1) {
                std::vector<T> result_data = {};
                for (size_t i = 0; i < shape.at(0); i++) {
                    for (size_t j = 0; j < shape.at(0); j++) {
                        if (j - i == k) {
                            result_data.push_back(data.at(i));
                        }
                        else {
                            result_data.push_back(0);
                        }
                    }
                }
                return Tensor<T>({ shape.at(0), shape.at(0) }, result_data);
            }
            else {
                std::vector<T> result_data = {};
                for (size_t i = 0; i < data.size(); i++) {
                    if ((i % shape.at(1)) - (i / shape.at(1)) == k) {
                        result_data.push_back(data.at(i));
                    }
                }
                return Tensor<T>(result_data);
            }
        }

        Tensor<T> clip(T a_min, T a_max) const {
            return transform([a_min, a_max](T a) {
                return a > a_max ? a_max : (a < a_min ? a_min : a);
                });
        }

        // Statistcal functions
        T sum() const {
            return std::accumulate(begin(data), end(data), 0, [](T x, T y) {
                return x + y;
                });
        }

        T prod() const {
            return std::accumulate(begin(data), end(data), 1, [](T x, T y) {
                return x * y;
                });
        }

        T mean() const {
            return sum() / data.size();
        }

        T average() const {
            return mean();
        }

        T average(const Tensor<T>& weights) const {
            return operator*(weights).sum() / weights.sum();
        }

        T var() const {
            return operator-(mean()).pow(2).sum();
        }

        T stdev() const {
            return std::sqrt(var());
        }

        T min() const {
            return std::min(data);
        }

        T max() const {
            return std::max(data);
        }

        T percentile(double q) const {
            return data.at(std::ceil(q / 100 * data.size()));
        }

        T quantile(double q) const {
            return data.at(std::ceil(q * data.size()));
        }

        T median() const {
            if(data.size() % 2 == 0) {
                return data.at(0);
            } else {
                return data
            }
        }

        // Other functions
        Tensor<T> copy() const {
            std::vector<size_t> cloned_shape = shape;
            std::vector<T> cloned_data = data;
            return Tensor<T>(cloned_shape, cloned_data);
        }


        Tensor<T> reshape(const std::vector<size_t>& new_shape) const {
            return Tensor<T>(new_shape, copy().data);
        }

        size_t ndim() const {
            return shape.size();
        }

        Tensor<T> transform(const std::function<T(T)>& func) const {
            std::vector<T> result_data;
            result_data = data;
            std::transform(result_data.begin(), result_data.end(), result_data.begin(), func);
            return Tensor<T>(shape, result_data);
        }

        bool bool_transform(const std::function<bool(T)>& func) const {
            std::vector<T> result_data;
            result_data = data;
            std::transform(result_data.begin(), result_data.end(), result_data.begin(), func);
            return Tensor<T>(shape, result_data);
        }

        Tensor<T> binary_transform(Tensor<T> other, const std::function<T(T, T)>& func) const {
            // error_check(data.size() == other.data.size(), 
            //     "operands are incompatible with " + std::to_string(data.size()) + " and " + std::to_string(other.data.size()) + " elements"); 
            std::vector<T> result_data;
            std::vector<size_t> indices;
            for (size_t i = 0; i < data.size(); i++) {
                indices.push_back(i);
            }
            std::transform(indices.begin(), indices.end(), std::back_inserter(result_data), [&](size_t i) {
                return func(data.at(i), other.data.at(i));
                });
            return Tensor<T>(copy().shape, result_data);
        }

        Tensor<T> binary_transform(T other, const std::function<T(T, T)>& func) const {
            std::vector<T> result_data = data;
            std::transform(result_data.begin(), result_data.end(), result_data.begin(), [func, other](T x) {
                return func(x, other);
                });
            return Tensor<T>(copy().shape, result_data);
        }

        Tensor<bool> bool_binary_transform(Tensor<T> other, const std::function<bool(T, T)>& func) const {
            // error_check(data.size() == other.data.size(), 
            //     "operands are incompatible with " + std::to_string(data.size()) + " and " + std::to_string(other.data.size()) + " elements");
            std::vector<bool> result_data;
            std::vector<size_t> indices;
            for (size_t i = 0; i < data.size(); i++) {
                indices.push_back(i);
            }
            std::transform(indices.begin(), indices.end(), std::back_inserter(result_data), [&](size_t i) {
                return func(data.at(i), other.data.at(i));
                });
            return Tensor<bool>(shape, result_data);
        }

        Tensor<bool> bool_binary_transform(T other, const std::function<bool(T, T)>& func) const {
            std::vector<bool> result_data;
            std::vector<size_t> indices;
            for (size_t i = 0; i < data.size(); i++) {
                indices.push_back(i);
            }
            std::transform(indices.begin(), indices.end(), std::back_inserter(result_data), [&](size_t i) {
                return func(data.at(i), other);
                });
            return Tensor<bool>(copy().shape, result_data);
        }

        Tensor<T> stack(const std::vector<Tensor<T> >& tensors) const {
            std::vector<size_t> result_shape = { tensors.size() + 1 };
            for (size_t i = 0; i < shape.size(); i++) {
                result_shape.push_back(shape.at(i));
            }
            std::vector<T> result_data;
            size_t data_size = data.size();
            for (size_t i = 0; i < data_size; i++) {
                result_data.push_back(data.at(i));
            }
            for (size_t i = 0; i < tensors.size(); i++) {
                for (size_t j = 0; j < data_size; j++) {
                    result_data.push_back(tensors.at(i).data.at(j));
                }
            }
            return Tensor<T>(result_shape, result_data);
        }

        Tensor<T> concatenate(const std::vector<Tensor<T> >& tensors) const {
            std::vector<size_t> result_shape = { shape.at(0) };
            for (size_t i = 0; i < tensors.size(); i++) {
                result_shape[0] += tensors.at(i).shape.at(0);
            }
            for (size_t i = 1; i < shape.size(); i++) {
                result_shape.push_back(shape.at(i));
            }
            std::vector<T> result_data;
            size_t data_size = data.size();
            for (size_t i = 0; i < data_size; i++) {
                result_data.push_back(data.at(i));
            }
            for (size_t i = 0; i < tensors.size(); i++) {
                for (size_t j = 0; j < data_size; j++) {
                    result_data.push_back(tensors.at(i).data.at(j));
                }
            }
            return Tensor<T>(result_shape, result_data);
        }

        Tensor<T> flip() const {
            std::vector<T> result_data(data);
            std::reverse(result_data.begin(), result_data.end());
            return Tensor<T>(copy().shape, result_data);
        }

        Tensor<T> at(const std::vector<size_t>& idx) const {
            size_t begin_idx = 0;
            for (size_t i = 0; i < idx.size(); i++) {
                size_t prod = 1;
                for (size_t j = i + 1; j < shape.size(); j++) {
                    prod *= shape.at(j);
                }
                begin_idx += idx.at(i) * prod;
            }
            // std::vector<size_t>::const_iterator result_shape_begin = shape.begin() + idx.size();
            // std::vector<size_t>::const_iterator result_shape_end = shape.end();
            std::vector<size_t> result_shape = { shape.begin() + idx.size(), shape.end() };
            size_t offset = get_prod(result_shape);
            // typename std::vector<T>::const_iterator result_data_begin = data.begin() + begin_idx;
            // typename std::vector<T>::const_iterator result_data_end = data.begin() + begin_idx + offset;
            std::vector<T> result_data = { data.begin() + begin_idx, data.begin() + begin_idx + offset };
            return Tensor<T>(result_shape, result_data);
        }

        void set(const std::vector<size_t>& idx, Tensor<T> val) {
            size_t begin_idx = 0;
            for (size_t i = 0; i < idx.size(); i++) {
                size_t prod = 1;
                for (size_t j = i + 1; j < shape.size(); j++) {
                    prod *= shape.at(j);
                }
                begin_idx += idx.at(i) * prod;
            }
            for (size_t i = 0; i < val.data.size(); i++) {
                data[begin_idx + i] = val.data[i];
            }
            // std::copy(data.begin() + begin_idx, data.begin() + begin_idx + val.data.size() + 1, val.data.begin());
        }

        T item() const {
            return data.at(0);
        }

        std::string str() const {
            if (data.size() == 0) {
                return "[]";
            }
            else if (ndim() == 0) {
                std::ostringstream oss;
                oss << item();
                return oss.str();
            }
            else if (ndim() == 1) {
                std::string result = "[";
                for (size_t i = 0; i < shape.at(0); i++) {
                    std::ostringstream oss;
                    oss << at(std::vector<size_t>{i}).item();
                    result = result + oss.str() + ", ";
                }
                result.pop_back();
                result.pop_back();
                result = result + "]";
                return result;
            }
            else {
                std::string result = "[";
                for (size_t i = 0; i < shape.at(0); i++) {
                    result = result + at(std::vector<size_t>{i}).str() + ", ";
                }
                result.pop_back();
                result.pop_back();
                result = result + "]";
                return result;
            }
        }

        Tensor<T> flatten() const {
            return Tensor<T>(copy().data);
        }

        Tensor<T> lstsq(const Tensor<T>& other) const {
            if (ndim() == 1 && other.ndim() == 1) {
                std::vector<T> a_data;
                for (size_t i = 0; i < data.size(); i++) {
                    a_data.push_back(data.at(i));
                    a_data.push_back(1.0);
                }
                Tensor<T> a({ shape.at(0), 2 }, a_data);
                Tensor<T> b({ shape.at(0), 1 }, other.copy().data);
                return a.lstsq(b);
            }
            else {
                Tensor<T> transpose = t();
                return transpose.matmul(*this).pinv().matmul(transpose.matmul(other)).flatten();
            }
        }

        Tensor<T> sort() const {
            std::vector<T> result_data = data;
            std::sort(result_data.begin(), result_data.end());
            return Tensor<T>(copy().shape, result_data);
        }

        Tensor<T> sort(const std::function<bool(T, T)>& comp) const {
            std::vector<T> result_data = data;
            std::sort(result_data.begin(), result_data.end(), comp);
            return Tensor<T>(copy().shape, result_data);
        }

        bool all() const {
            return std::all_of(
                std::begin(data),
                std::end(data),
                [](T a) {
                    return a != 0;
                });
        }

        bool any() const {
            return !std::all_of(
                std::begin(data),
                std::end(data),
                [](T a) {
                    return a == 0;
                });
        }

        Tensor<T> roll(int shift) const {
            size_t data_size = data.size();
            std::vector<T> result_data;
            for(size_t i = 0; i < data_size; i++) {
                int idx = ((int)i - shift) % (int)data_size;
                if(idx < 0) {
                    idx += data_size;
                }
                result_data.push_back(data[idx]);
            }
            return Tensor<T>(copy().shape, result_data);
        }

        Tensor<T> repeat(size_t repeats) const {
            std::vector<T> result_data;
            for(size_t i = 0; i < data.size(); i++) {
                for(size_t j = 0; j < repeats; j++) {
                    result_data.push_back(data.at(i));
                }
            }
            return Tensor<T>(result_data);
        }

        Tensor<T> subtensor(const std::vector<size_t>& begin, const std::vector<size_t>& end) const {
            std::vector<size_t> result_shape = shape;
            size_t begin_idx = 0;
            for (size_t i = 0; i < begin.size(); i++) {
                size_t prod = 1;
                for (size_t j = i + 1; j < shape.size(); j++) {
                    prod *= shape.at(j);
                }
                size_t temp = begin.at(i);
                begin_idx += temp * prod;
                result_shape[i] = end.at(i) - temp;
            }
            size_t end_idx = 0;
            for (size_t i = 0; i < end.size(); i++) {
                size_t prod = 1;
                for (size_t j = i + 1; j < shape.size(); j++) {
                    prod *= shape.at(j);
                }
                end_idx += end.at(i) * prod;
            }
            
            std::vector<T> result_data = { data.begin() + begin_idx, data.begin() + end_idx };
            return Tensor<T>(result_shape, result_data);
        }

        // Static functions
        static Tensor<T> empty(const std::vector<size_t>& shape) {
            std::vector<T> result_data(get_prod(shape));
            return Tensor<T>(shape, result_data);
        }

        static Tensor<T> empty_like(const Tensor<T>& tensor) {
            return empty(tensor.shape);
        }

        static Tensor<T> ones(const std::vector<size_t>& shape) {
            std::vector<T> result_data(get_prod(shape), 1);
            return Tensor<T>(shape, result_data);
        }

        static Tensor<T> ones_like(const Tensor<T>& tensor) {
            return ones(tensor.shape);
        }

        static Tensor<T> zeros(const std::vector<size_t>& shape) {
            std::vector<T> result_data(get_prod(shape), 0);
            return Tensor<T>(shape, result_data);
        }

        static Tensor<T> zeros_like(const Tensor<T>& tensor) {
            return zeros(tensor.shape);
        }

        static Tensor<T> full(const std::vector<size_t>& shape, T fill_value) {
            std::vector<T> result_data(get_prod(shape), fill_value);
            return Tensor<T>(shape, result_data);
        }

        static Tensor<T> full_like(const Tensor<T>& tensor, T fill_value) {
            return zeros(tensor.shape, fill_value);
        }


        static Tensor<T> random(const std::vector<size_t>& shape) {
            std::random_device rd;
            std::mt19937 mersenne_engine(rd());
            std::uniform_real_distribution<double> dist(0, 1); // range [0, 1)
            size_t result_data_size = get_prod(shape);
            std::vector<T> result_data;
            for (size_t i = 0; i < result_data_size; i++) {
                result_data.push_back(dist(rd));
            };
            return Tensor<T>(shape, result_data);
        }

        static Tensor<T> uniform(T low, T high, const std::vector<size_t>& shape) {
            std::random_device rd;
            std::mt19937 mersenne_engine(rd());
            std::uniform_real_distribution<long double> dist((long double)low, (long double)high); // range [low, high)
            size_t result_data_size = get_prod(shape);
            std::vector<T> result_data;
            for (size_t i = 0; i < result_data_size; i++) {
                result_data.push_back(dist(rd));
            };
            return Tensor<T>(shape, result_data);
        }

        static Tensor<T> eye(size_t size) {
            std::vector<T> result_data(size * size, 0);
            for (size_t i = 0; i < size; i++) {
                result_data[i * size + i] = 1;
            }
            return Tensor<T>({ size, size }, result_data);
        }

        static Tensor<T> tri(size_t n, size_t m = -1, int k = 0) {
            if (m == -1) {
                m = n;
            }
            std::vector<T> result_data(k);
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < m; j++) {
                    // std::cout << i << ' ' << j << ' ' << i * m + j << '\n';
                    if (j < i + k) {
                        result_data[i * m + j] = 1;
                    }
                    else {
                        result_data[i * m + j] = 0;
                    }
                }
            }

            return Tensor<T>({ n, m }, result_data);
        }

        static Tensor<T> arange(T start, T stop, T step = 1) {
            std::vector<T> result_data;
            for (T i = start; i < stop; i += step) {
                result_data.push_back(i);
            }
            return Tensor<T>(result_data);
        }

        static Tensor<T> arangez(T stop, T step = 1) {
            return arange(0, stop, step);
        }

        static Tensor<T> linspace(T start, T stop, T num = 50, bool endpoint = true) {
            T step = (stop - start) / (num - 1);
            if (!endpoint) {
                step = (stop - start) / (num);
            }
            std::vector<T> result_data;
            for (T i = start; i < stop; i += step) {
                result_data.push_back(i);
            }
            if (endpoint) {
                result_data.push_back(stop);
            }
            return Tensor<T>(result_data);
        }

        static Tensor<T> logspace(T start, T stop, T num = 50, bool endpoint = true, T base = 10) {
            return linspace(start, stop, num, endpoint).transform([base](T a) {
                return std::pow(base, a);
                });
        }

        static Tensor<T> geomspace(T start, T stop, T num = 50, bool endpoint = true, T base = 10) {
            return logspace(std::log(start) / std::log(base), std::log(stop) / std::log(base), num, endpoint, base);
        }
    protected:
        // Utillity functions
        static size_t get_prod(const std::vector<size_t>& vec) {
            return std::accumulate(begin(vec), end(vec), 1, std::multiplies<size_t>());
        }

        void error_check(bool cond, const std::string &msg) const {
            if(!cond) {
                throw TPPException(msg);
            }
        }
    };
}