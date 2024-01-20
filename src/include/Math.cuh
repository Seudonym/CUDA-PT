#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__host__ __device__ struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3(float x) : x(x), y(x), z(x) {}

    __host__ __device__ Vec3 operator+(const Vec3 &v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3 &v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3 operator*(const Vec3 &v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__ Vec3 operator/(const Vec3 &v) const {
        return Vec3(x / v.x, y / v.y, z / v.z);
    }

    __host__ __device__ void print() const {
        printf("(%f, %f, %f)", x, y, z);
    }
};

__host__ __device__ Vec3 operator*(const Vec3 &v, float f) { return Vec3(v.x * f, v.y * f, v.z * f); }
__host__ __device__ Vec3 operator*(float f, const Vec3 &v) { return Vec3(v.x * f, v.y * f, v.z * f); }
__host__ __device__ Vec3 operator/(const Vec3 &v, float f) { return Vec3(v.x / f, v.y / f, v.z / f); }
__host__ __device__ Vec3 operator/(float f, Vec3 &v) { return Vec3(v.x / f, v.y / f, v.z / f); }

__host__ __device__ float dot(const Vec3 &v1, const Vec3 &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ float length(const Vec3 &v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ Vec3 normalize(const Vec3 &v) {
    return v / sqrtf(dot(v, v));
}

__host__ __device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ Vec3 refract(const Vec3& v, const Vec3& n, float ni_over_nt) {
    float cosTheta = fminf(dot(-1.0f * v, n), 1.0f);
    Vec3 rOutPerp = ni_over_nt * (v + cosTheta * n);
    Vec3 rOutParallel = -sqrtf(fabsf(1.0f - dot(rOutPerp, rOutPerp))) * n;
    return rOutParallel + rOutPerp;
}

__host__ __device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 *= r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__host__ __device__ struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3 &origin, const Vec3 &direction) : origin(origin), direction(direction) {}

    __host__ __device__ Vec3 pointAt(float t) const {
        return origin + direction * t;
    }
};