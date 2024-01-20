#pragma once

#include <cuda_runtime.h>

#include <Math.cuh>
#include <Random.cuh>
#include <Solid.cuh>

class Material {
public:
    __device__ virtual bool scatter(const Ray &ray, const HitRecord &record, Vec3 &attenuation, Ray &scattered, curandState *state) const = 0;
};

class Lambertian : public Material {
    Vec3 albedo;

public:
    __device__ Lambertian(const Vec3 &albedo) : albedo(albedo) {}

    __device__ virtual bool scatter(const Ray &ray, const HitRecord &record, Vec3 &attenuation, Ray &scattered, curandState *state) const override {
        Vec3 target = record.p + record.normal + randomUnitVector(state);
        attenuation = albedo;
        scattered = Ray(record.p, target - record.p);
        return true;
    }
};

class Metal : public Material {
    Vec3 albedo;
    float fuzz;

public:
    __device__ Metal(const Vec3 &albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

    __device__ virtual bool scatter(const Ray& ray, const HitRecord& record, Vec3 &attenuation, Ray& scattered, curandState *state) const override {
        Vec3 reflected = reflect(ray.direction, record.normal);
        scattered = Ray(record.p, reflected + fuzz * randomUnitVector(state));
        attenuation = albedo;
        return (dot(scattered.direction, record.normal) > 0);
    }
};