#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#include <Math.cuh>
#include <Random.cuh>
#include <Camera.cuh>
#include <Solid.cuh>
#include <World.cuh>
#include <Material.cuh>

#include <SDL2/SDL.h>

#include <stdio.h>
#include <float.h>

const int width = 1024;
const int height = 768;

const float aspect = float(width) / float(height);

__global__ void createWorld(Solid **list, Solid **world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        list[0] = new Sphere(Vec3(-0.6f, 0.0f, -1.0f), 0.5f, new Lambertian(Vec3(0.5f)));
        list[1] = new Sphere(Vec3(+0.6f, 0.0f, -1.0f), 0.5f, new Dielectric(1.5f));
        list[2] = new Sphere(Vec3(0.0f, -100.5f, -1.0f), 100.0f, new Lambertian(Vec3(0.8f, 0.8f, 0.4f)));
        *world = new World(list, 3);
    }
}

__device__ Vec3 traceRay(Ray ray, Solid **world, curandState *state) {
    Ray currentRay = ray;
    Vec3 currentAttenuation = Vec3(1.0f);

    for (int i = 0; i < 50; i++) {
        HitRecord record;
        if ((*world)->hit(currentRay, 0.001f, FLT_MAX, record)) {
            Ray scattered;
            Vec3 attenuation;
            if (record.material->scatter(currentRay, record, attenuation, scattered, state)) {
                currentRay = scattered;
                currentAttenuation = currentAttenuation * attenuation;
            } else
                break;
        } else {
            Vec3 unitDirection = normalize(currentRay.direction);
            // float t = 0.5f * (unitDirection.x + 1.0f); Vec3 sky = (t < 0.5f) ? Vec3(1.0f, 0.0f, 0.0f) : Vec3(0.0f, 0.0f, 1.0f);
            float t = 0.5f * (unitDirection.y + 1.0f);  Vec3 sky = (1.0f - t) * Vec3(1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);

            return currentAttenuation * sky;
        }
    }
    return Vec3();
}

__global__ void renderKernel(Camera *camera, Solid **world, int numSamples, uint8_t *frameBuffer, curandState *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height))
        return;

    int index = i + j * width;
    curandState localState = state[index];
    Vec3 color;

    for (int s = 0; s < numSamples; s++) {
        float x = float(i) + curand_uniform(&localState) - 0.5f;
        float y = float(j) + curand_uniform(&localState) - 0.5f;
        x = 2.0f * (x / float(width)) - 1.0f;
        y = 2.0f * (y / float(height)) - 1.0f;
        y *= -1.0f;

        Ray ray = camera->getRay(x, y);

        color = color + traceRay(ray, world, &localState);
    }
    color = color / float(numSamples);

    frameBuffer[(i + j * width) * 3 + 0] = color.x * 255;
    frameBuffer[(i + j * width) * 3 + 1] = color.y * 255;
    frameBuffer[(i + j * width) * 3 + 2] = color.z * 255;
}

int main() {
    uint8_t *frameBuffer;
    Camera *camera;
    Solid **world;
    Solid **list;
    curandState *state;
    int numSamples = 1;

    cudaMallocManaged(&list, sizeof(Solid *) * 2);
    cudaMallocManaged(&world, sizeof(Solid *));

    createWorld<<<1, 1>>>(list, world);

    cudaMallocManaged(&state, width * height * sizeof(curandState));

    dim3 blocks(width / 16 + 1, height / 16 + 1);
    dim3 threads(16, 16);

    initRandom<<<blocks, threads>>>(width, height, state);

    cudaMallocManaged(&frameBuffer, width * height * 3);
    cudaMallocManaged(&camera, sizeof(Camera));

    camera->position = Vec3(0.0f, 0.0f, 1.0f);
    camera->aspectRatio = aspect;
    camera->fov = 3.14f / 2.0f;
    camera->lookAt = Vec3(0.0f, 0.0f, -1.0f);
    camera->up = Vec3(0.0f, 1.0f, 0.0f);

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("Path Tracing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
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
                // MOVE CAMERA
                if (event.key.keysym.sym == SDLK_w) {
                    Vec3 direction = camera->lookAt - camera->position;
                    direction = normalize(direction);
                    camera->position = camera->position + direction * 0.1f;
                    camera->lookAt = camera->lookAt + direction * 0.1f;
                }
                if (event.key.keysym.sym == SDLK_s) {
                    Vec3 direction = camera->lookAt - camera->position;
                    direction = normalize(direction);
                    camera->position = camera->position - direction * 0.1f;
                    camera->lookAt = camera->lookAt - direction * 0.1f;
                }
                if (event.key.keysym.sym == SDLK_d) {
                    Vec3 direction = camera->lookAt - camera->position;
                    Vec3 right = cross(camera->up, direction);
                    right = normalize(right);
                    camera->position = camera->position - right * 0.1f;
                    camera->lookAt = camera->lookAt - right * 0.1f;
                }
                if (event.key.keysym.sym == SDLK_a) {
                    Vec3 direction = camera->lookAt - camera->position;
                    Vec3 right = cross(camera->up, direction);
                    right = normalize(right);
                    camera->position = camera->position + right * 0.1f;
                    camera->lookAt = camera->lookAt + right * 0.1f;
                }

                // ROTATE CAMERA
                if (event.key.keysym.sym == SDLK_LEFT) {
                    Vec3 direction = camera->lookAt - camera->position;
                    float theta = -0.1f;
                    float x = direction.x;
                    float z = direction.z;
                    direction.x = x * cos(theta) - z * sin(theta);
                    direction.z = x * sin(theta) + z * cos(theta);
                    camera->lookAt = camera->position + direction;
                }
                if (event.key.keysym.sym == SDLK_RIGHT) {
                    Vec3 direction = camera->lookAt - camera->position;
                    float theta = 0.1f;
                    float x = direction.x;
                    float z = direction.z;
                    direction.x = x * cos(theta) - z * sin(theta);
                    direction.z = x * sin(theta) + z * cos(theta);
                    camera->lookAt = camera->position + direction;
                }
                if (event.key.keysym.sym == SDLK_UP) {
                    Vec3 direction = camera->lookAt - camera->position;
                    float theta = 0.1f;
                    float y = direction.y;
                    float z = direction.z;
                    direction.y = y * cos(theta) - z * sin(theta);
                    direction.z = y * sin(theta) + z * cos(theta);
                    camera->lookAt = camera->position + direction;
                }
                if (event.key.keysym.sym == SDLK_DOWN) {
                    Vec3 direction = camera->lookAt - camera->position;
                    float theta = -0.1f;
                    float y = direction.y;
                    float z = direction.z;
                    direction.y = y * cos(theta) - z * sin(theta);
                    direction.z = y * sin(theta) + z * cos(theta);
                    camera->lookAt = camera->position + direction;
                }

                if (event.key.keysym.sym == SDLK_q) {
                    numSamples--;
                }
                if (event.key.keysym.sym == SDLK_e) {
                    numSamples++;
                }
            }
        }

        if (render) {
            renderKernel<<<blocks, threads>>>(camera, world, numSamples, frameBuffer, state);
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
