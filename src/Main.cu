#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Math.cuh>
#include <Solid.cuh>
#include <World.cuh>
#include <Camera.cuh>

#include <SDL2/SDL.h>

#include <stdio.h>

const int width = 1024;
const int height = 768;

const float aspect = float(width) / float(height);

__global__ void createWorld(Solid** list, Solid** world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        list[0] = new Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f);
        list[1] = new Sphere(Vec3(0.0f, -100.5f, -1.0f), 100.0f);
        *world = new World(list, 2);
    }
}

__global__ void renderKernel(Camera *camera, Solid** world, uint8_t *frameBuffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height))
        return;

    float x = float(i) * 2.0f / float(width) - 1.0f;
    float y = float(j) * 2.0f / float(height) - 1.0f;
    y *= -1.0f;
    Vec3 color;

    Ray ray = camera->getRay(x, y);

    HitRecord record;
    if ((*world)->hit(ray, 0.0f, 1000.0f, record)) {
        color = (record.normal * 0.5f + 0.5f) * 255;
        frameBuffer[(i + j * width) * 3 + 0] = color.x;
        frameBuffer[(i + j * width) * 3 + 1] = color.y;
        frameBuffer[(i + j * width) * 3 + 2] = color.z;
        return;
    }

    Vec3 unitDirection = normalize(ray.direction);
    float t = 0.5f * (unitDirection.y + 1.0f);
    color = (1.0f - t) * Vec3(1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);

    frameBuffer[(i + j * width) * 3 + 0] = color.x * 255;
    frameBuffer[(i + j * width) * 3 + 1] = color.y * 255;
    frameBuffer[(i + j * width) * 3 + 2] = color.z * 255;
}

int main() {
    uint8_t *frameBuffer;
    Camera *camera;
    Solid** world;
    Solid** list;

    cudaMallocManaged(&list, sizeof(Solid*) * 2);
    cudaMallocManaged(&world, sizeof(Solid*));

    createWorld<<<1, 1>>>(list, world);

    cudaMallocManaged(&frameBuffer, width * height * 3);
    cudaMallocManaged(&camera, sizeof(Camera));

    camera->position = Vec3(0.0f, 0.0f, 1.0f);
    camera->aspectRatio = aspect;
    camera->fov = 3.14f / 2.0f;
    camera->lookAt = Vec3(0.0f, 0.0f, -1.0f);
    camera->up = Vec3(0.0f, 1.0f, 0.0f);

    dim3 blocks(width / 16 + 1, height / 16 + 1);
    dim3 threads(16, 16);

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("Ray Tracing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, width, height);
    SDL_Event event;
    bool running = true;
    bool render = true;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            if (event.type == SDL_KEYDOWN) {
                render = true;
                if (event.key.keysym.sym == SDLK_a){
                    camera->position.x -= 0.1f;
                    camera->lookAt.x -= 0.1f;
                }
                if (event.key.keysym.sym == SDLK_d){
                    camera->position.x += 0.1f;
                    camera->lookAt.x += 0.1f;
                }
                if (event.key.keysym.sym == SDLK_w){
                    camera->position.y += 0.1f;
                    camera->lookAt.y += 0.1f;
                }
                if (event.key.keysym.sym == SDLK_s){
                    camera->position.y -= 0.1f;
                    camera->lookAt.y -= 0.1f;
                }
            }
        }

        if (render) {
            renderKernel<<<blocks, threads>>>(camera, world, frameBuffer);
            cudaDeviceSynchronize();
            render = false;
        }

        SDL_UpdateTexture(texture, NULL, frameBuffer, width * 3);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    cudaFree(frameBuffer);

    return 0;
}
