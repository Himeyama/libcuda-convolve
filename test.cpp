#include <iostream>
#include <vector>
#include <cuda-convolve.hpp>

int main(){
    std::vector<float> a{1, 2, 3};
    std::vector<float> v{0, 1, 0.5};
    std::vector<float> conv = convolve(a, v);
    for(int i = 0; i < conv.size(); i++){
        std::cout << conv[i] << std::endl;
    }

    std::vector<double> da{1, 2, 3};
    std::vector<double> dv{0, 1, 0.5};
    std::vector<double> dconv = convolve(da, dv);
    for(int i = 0; i < dconv.size(); i++){
        std::cout << dconv[i] << std::endl;
    }

    return 0;
}
