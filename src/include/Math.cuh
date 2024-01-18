#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__host__ __device__ struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3(float x) : x(x), y(x), z(x) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3 operator*(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__ Vec3 operator/(const Vec3& v) const {
        return Vec3(x / v.x, y / v.y, z / v.z);
    }

    __host__ __device__ Vec3 operator*(float f) const {
        return Vec3(x * f, y * f, z * f);
    }

    __host__ __device__ Vec3 operator/(float f) const {
        return Vec3(x / f, y / f, z / f);
    }

    __host__ __device__ void print() const {
        printf("(%f, %f, %f)", x, y, z);
    }
};

__host__ __device__ float dot(const Vec3& v1, const Vec3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ Vec3 cross(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ Vec3 normalize(const Vec3& v) {
    return v / sqrtf(dot(v, v));
}


__host__ __device__ struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}

    __host__ __device__ Vec3 pointAt(float t) const {
        return origin + direction * t;
    }
};