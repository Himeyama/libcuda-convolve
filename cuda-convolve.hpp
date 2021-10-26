#ifndef CUDA_CONVOLVE_HPP
#define CUDA_CONVOLVE_HPP

template <typename T>
std::vector<T> convolve(std::vector<T> a, std::vector<T> v);
template <typename T>
std::vector<T> convolve_same(std::vector<T> a, std::vector<T> v);
template <typename T>
std::vector<T> convolve_valid(std::vector<T> a, std::vector<T> v);

#endif