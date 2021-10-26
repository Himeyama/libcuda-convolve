#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#define max(a, b) ((a > b) ? a : b)

template <typename T>
__global__
void cuda_convolve_full(T *a, T *v, T *conv, long a_size, long v_size){
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    conv[i] = 0;
    if(i < a_size + v_size - 1)
        for(long j = 0; j < v_size; j++)
            conv[i] += v[j] * ((i - j >= a_size || i - j < 0) ? 0 : a[i - j]);
}

template <typename T>
T* fary2cuda(std::vector<T> a){
    T *g;
    cudaError_t err = cudaMalloc((void**)&g, sizeof(T) * a.size());
    if(err) exit(err);
    cudaMemcpy(g, a.data(), sizeof(T) * a.size(), cudaMemcpyHostToDevice);
    return g;
}

template <typename T>
std::vector<T> convolve(std::vector<T> a, std::vector<T> v){
    std::vector<T> conv(a.size() + v.size() - 1);
    T* ga = fary2cuda(a);
    T* gv = fary2cuda(v);
    T* gconv = fary2cuda(conv);
    cuda_convolve_full<<<(a.size()+256)/256, 256>>>(ga, gv, gconv, a.size(), v.size());
    cudaDeviceSynchronize();
    cudaMemcpy(conv.data(), gconv, sizeof(T) * conv.size(), cudaMemcpyDeviceToHost);
    cudaFree(ga);
    cudaFree(gv);
    cudaFree(gconv);
    cudaDeviceReset();
    return conv;
}

template <typename T>
std::vector<T> convolve_same(std::vector<T> a, std::vector<T> v){
    std::vector<T> conv(max(a.size(), v.size()));
    long sidx = (int)round(v.size() / 2.0) - 1;
    std::vector<T> conv_full = convolve(a, v);
    memcpy(conv.data(), conv_full.data() + sidx, sizeof(T) * conv.size());
    return conv;
}

template <typename T>
std::vector<T> convolve_valid(std::vector<T> a, std::vector<T> v){
    std::vector<T> conv(max(a.size(), v.size()) - min(a.size(), v.size()) + 1);
    long sidx = v.size() - 1;
    std::vector<T> conv_full = convolve(a, v);
    memcpy(conv.data(), conv_full.data() + sidx, sizeof(T) * conv.size());
    return conv;
}

template std::vector<float> convolve<float>(std::vector<float>, std::vector<float>);
template std::vector<float> convolve_valid<float>(std::vector<float>, std::vector<float>);
template std::vector<float> convolve_same<float>(std::vector<float>, std::vector<float>);

template std::vector<double> convolve<double>(std::vector<double>, std::vector<double>);
template std::vector<double> convolve_valid<double>(std::vector<double>, std::vector<double>);
template std::vector<double> convolve_same<double>(std::vector<double>, std::vector<double>);
