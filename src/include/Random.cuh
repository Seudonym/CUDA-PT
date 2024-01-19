#pragma once

#include <curand.h>
#include <curand_kernel.h>

__global__ void initRandom(int width, int height, curandState* state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(1234, idx, 0, &state[idx]);
}