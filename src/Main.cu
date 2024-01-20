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
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <stdio.h>
#include <float.h>

const int width = 1024;
const int height = 768;

const float aspect = float(width) / float(height);

__global__ void createWorld(Solid **list, Solid **world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        list[0] = new Sphere(Vec3(-0.6f, 0.0f, -1.0f), 0.5f, new Lambertian(Vec3(0.5f)));
        list[1] = new Sphere(Vec3(+0.6f, 0.0f, -1.0f), 0.5f, new Dielectric(1.5f));
        list[2] = new Sphere(Vec3(+1.8f, 0.0f, -1.0f), 0.5f, new Metal(Vec3(0.0f, 1.0f, 1.0f), 0.4f));
        *world = new World(list, 3);
    }
}

__device__ Vec3 traceRay(Ray ray, Solid **world, float *image, curandState *state) {
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
            float u = (0.5 + atan2(unitDirection.x, unitDirection.z) / (2.0 * M_PIf));
            float v = (0.5 - asin(unitDirection.y) / M_PIf);

            int x = (int)(u * (float)1024);
            int y = (int)(v * (float)512);
            int index = 3 * (y * 1024 + x);
            if (index < 0 || index > 1024 * 512 * 3) return Vec3(1.0f);

            Vec3 sky = Vec3(image[index + 0], image[index + 1], image[index + 2]);

            return currentAttenuation * sky;
        }
    }
    return Vec3();
}

__global__ void renderKernel(Camera *camera, Solid **world, int numSamples, uint8_t *frameBuffer, float *image, curandState *state) {
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

        color = color + traceRay(ray, world, image, &localState);
    }
    color = color / float(numSamples);

    frameBuffer[(i + j * width) * 3 + 0] = color.x * 255;
    frameBuffer[(i + j * width) * 3 + 1] = color.y * 255;
    frameBuffer[(i + j * width) * 3 + 2] = color.z * 255;
}

int main() {
    uint8_t *frameBuffer;
    float *image;
    Camera *camera;
    Solid **world;
    Solid **list;
    curandState *state;
    int numSamples = 1;

    dim3 blocks(width / 16 + 1, height / 16 + 1);
    dim3 threads(16, 16);

    cudaMallocManaged(&list, sizeof(Solid *) * 3);
    cudaMallocManaged(&world, sizeof(Solid *));
    createWorld<<<1, 1>>>(list, world);
    cudaDeviceSynchronize();

    cudaMallocManaged(&state, width * height * sizeof(curandState));
    initRandom<<<blocks, threads>>>(width, height, state);
    cudaDeviceSynchronize();

    int imgWidth, imgHeight, channels;
    unsigned char *img = stbi_load("./hdri/hdri1.hdr", &imgWidth, &imgHeight, &channels, 0);
    cudaMallocManaged(&image, imgWidth * imgHeight * channels * sizeof(float));
    for (int i = 0; i < imgWidth * imgHeight * channels; i++) {
        float value = static_cast<float>(img[i]);
        image[i] = value / 255.0f;
    }
    stbi_image_free(img);

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
            renderKernel<<<blocks, threads>>>(camera, world, numSamples, frameBuffer, image, state);
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
