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

    __device__ virtual bool scatter(const Ray &ray, const HitRecord &record, Vec3 &attenuation, Ray &scattered, curandState *state) const override {
        Vec3 reflected = reflect(ray.direction, record.normal);
        scattered = Ray(record.p, reflected + fuzz * randomUnitVector(state));
        attenuation = albedo;
        return (dot(scattered.direction, record.normal) > 0);
    }
};

class Dielectric : public Material {
    float refractionIndex;
    Vec3 albedo;

public:
    __device__ Dielectric(float refractionIndex) : refractionIndex(refractionIndex) {}

    __device__ virtual bool scatter(const Ray &ray, const HitRecord &record, Vec3 &attenuation, Ray &scattered, curandState *state) const override {
        attenuation = Vec3(1.0f, 1.0f, 1.0f);
        float eta = record.frontFace ? (1.0f / refractionIndex) : refractionIndex;

        Vec3 unitDirection = normalize(ray.direction);

        float cosTheta = fminf(dot(-1.0f * unitDirection, record.normal), 1.0f);
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

        bool cannotRefract = eta * sinTheta > 1.0f;
        Vec3 direction;

        if (cannotRefract || schlick(cosTheta, eta) > curand_uniform(state)) {
            direction = reflect(unitDirection, record.normal);
        } else {
            direction = refract(unitDirection, record.normal, eta);
        }

        scattered = Ray(record.p, direction);

        return true;
    }
};