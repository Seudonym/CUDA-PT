#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Math.cuh>
#include <Solid.cuh>
#include <Camera.cuh>

#include <stdio.h>

const int width = 1024;
const int height = 768;

const float aspect = float(width) / float(height);

__global__ void renderKernel(Vec3 *frameBuffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height))
        return;

    float x = float(i) * 2.0f / float(width) - 1.0f;
    float y = float(j) * 2.0f / float(height) - 1.0f;

    Camera camera;
    camera.position = Vec3(0.0f, 0.0f,1.0f);
    camera.aspectRatio = aspect;
    camera.fov = 3.14f / 2.0f;
    camera.lookAt = Vec3(0.0f, 0.0f, -1.0f);
    camera.up = Vec3(0.0f, 1.0f, 0.0f);
    Ray ray = camera.getRay(x, y);
    
    Sphere sphere(Vec3(0.0f, 0.0f, -1.0f), 1.0f);
    HitRecord record;
    if (sphere.hit(ray, 0.0f, 1000.0f, record)) {
        frameBuffer[i + j * width] = Vec3(1.0f, 0.0f, 0.0f);
        return;
    }

    Vec3 unitDirection = normalize(ray.direction);
    float t = 0.5f * (unitDirection.y + 1.0f);
    Vec3 color = (1.0f - t) * Vec3(1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);

    frameBuffer[i + j * width] = color;
}

int main() {
    Vec3 *frameBuffer;
    cudaMallocManaged(&frameBuffer, width * height * sizeof(Vec3));

    dim3 blocks(width / 16 + 1, height / 16 + 1);
    dim3 threads(16, 16);

    renderKernel<<<blocks, threads>>>(frameBuffer);
    cudaDeviceSynchronize();

    FILE *file = fopen("image.ppm", "w");
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
