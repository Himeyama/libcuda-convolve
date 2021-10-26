#include <iostream>
#include <vector>
#include <cuda-convolve.hpp>

int main(){
    std::vector<float> a(3);
    for(int i = 0; i < a.size(); i++)
        a[i] = i + 1;
    std::vector<float> v(3);
    v[0] = 0;
    v[1] = 1;
    v[2] = 0.5;

    std::vector<float> conv = convolve(a, v);
    for(int i = 0; i < conv.size(); i++){
        std::cout << conv[i] << std::endl;
    }

    return 0;
}
