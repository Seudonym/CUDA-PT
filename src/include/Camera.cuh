#pragma once

#include <cuda_runtime.h>
#include <Math.cuh>

struct Camera {
    Vec3 position;
    Vec3 lookAt;
    Vec3 up;
    float fov;
    float aspectRatio;

    __device__ Ray getRay(float u, float v) const {
        Vec3 w = normalize(lookAt - position);
        Vec3 uVec = normalize(cross(w, up));
        Vec3 vVec = cross(uVec, w);

        float halfHeight = tan(fov / 2);
        float halfWidth = aspectRatio * halfHeight;

        Vec3 origin = position;
        Vec3 direction = normalize((uVec * halfWidth * u) + (vVec * halfHeight * v) + w);
        return Ray(origin, direction);
    }
};