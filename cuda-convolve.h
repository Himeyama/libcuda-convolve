/*
 *  (c) 2021 Murata Mitsuharu
 *  Licensed under the MIT License.
 *  source: https://github.com/example-cuda
*/

#include "float-vector.h"

#ifndef CUDA_CONVOLVE_H
#define CUDA_CONVOLVE_H

FloatVec convolve(FloatVec a, FloatVec v);
FloatVec convolve_same(FloatVec a, FloatVec v);
FloatVec convolve_valid(FloatVec a, FloatVec v);

#endif