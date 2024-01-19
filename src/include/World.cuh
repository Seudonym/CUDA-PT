#pragma once

#include <Solid.cuh>
#include <vector>

class World : public Solid {
public:
    Solid** list;
    unsigned size;

    __host__ __device__ World() {
        list = new Solid*[1];
        size = 0;
    }
    __host__ __device__ World(Solid ** list, unsigned size) {
        this->list = list;
        this->size = size;
    }

    __device__ virtual bool hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const override {
        HitRecord tempRecord;
        bool hitAnything = false;
        float closestSoFar = tMax;

        for (unsigned i = 0; i < size; i++) {
            if (list[i]->hit(ray, tMin, closestSoFar, tempRecord)) {
                hitAnything = true;
                closestSoFar = tempRecord.t;
                record = tempRecord;
            }
        }

        return hitAnything;
    }

    __device__ void add(Solid *solid) {
        list[0] = solid;
    }
};