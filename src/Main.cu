#include <Math.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

const int width = 1024;
const int height = 768;

__global__ void renderKernel(Vec3* frameBuffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height)) return;

    Vec3 color = {float(i) / width, float(j) / height, 0.2f};
    
    frameBuffer[i + j * width] = color;
}

int main() {
    Vec3* frameBuffer;
    cudaMallocManaged(&frameBuffer, width * height * sizeof(Vec3));

    dim3 blocks(width / 16 + 1, height / 16 + 1);
    dim3 threads(16, 16);

    renderKernel<<<blocks, threads>>>(frameBuffer);
    cudaDeviceSynchronize();

    FILE* file = fopen("image.ppm", "w");
    fprintf(file, "P3\n%d %d\n%d\n", width, height, 255);
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            Vec3 color = frameBuffer[i + j * width];
            int ir = int(255.99 * color.x);
            int ig = int(255.99 * color.y);
            int ib = int(255.99 * color.z);
            fprintf(file, "%d %d %d\n", ir, ig, ib);
        }
    }

    cudaFree(frameBuffer);


    return 0;
}
