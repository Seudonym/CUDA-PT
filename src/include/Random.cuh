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

__device__ Vec3 randomUnitVector(curandState* state) {
    float a = curand_uniform(state) * 2 * M_PI;
    float z = curand_uniform(state) * 2 - 1;
    float r = sqrt(1 - z * z);
    return Vec3(r * cos(a), r * sin(a), z);
}