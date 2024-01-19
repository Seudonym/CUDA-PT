#pragma once

#include <cuda_runtime.h>
#include <Math.cuh>

__device__ struct HitRecord {
    float t;
    Vec3 p;
    Vec3 normal;
};

__device__ class Solid {
public:
    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
};

__device__ class Sphere : public Solid {
    Vec3 center;
    float radius;

public:
    __host__ __device__ Sphere() : radius(1.0f) {}
    __host__ __device__ Sphere(const Vec3 &center, float radius) : center(center), radius(radius) {}

    __device__ bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override {
        Vec3 oc = r.origin - center;
        float a = dot(r.direction, r.direction);
        float b = dot(oc, r.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.pointAt(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.pointAt(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
        }
        return false;
    }
};