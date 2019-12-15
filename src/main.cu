#include <stdio.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

#include "cutil_math.h"

#define width 512
#define height 512
#define samples 128

struct Ray {
    float3 origin;
    float3 direction;

    __host__ __device__
    Ray(float3 o_, float3 d_)
        : origin { o_ }
        , direction { d_ }
    { }

    __host__ __device__
    float3 at(float distance) const {
        return this->origin + this->direction * distance;
    }
};

enum Material_t {
    DIFF,
    SPEC,
    REFR,
};

class Sphere;

struct Intersection {
    bool valid = false;

    const Sphere* object = nullptr;

    float distance = 0;
    float3 point;
    float3 normal;
};

struct Sphere {
    float radius;
    float3 position;
    float3 emission;
    float3 colour;
    Material_t material;

    __device__
    Intersection intersect(const Ray& r) const {
        Intersection intersection {};

        intersection.object = this;

        float3 op = this->position - r.origin;
        float epsilon = 0.001f;
        float b = dot(op, r.direction);

        float disc = b * b - dot(op, op) + this->radius * this->radius;
        if (disc < 0) {
            return intersection;
        }
        else {
            disc = sqrtf(disc);
        }

        float x0 = b - disc;
        float x1 = b + disc;

        if (x0 > epsilon) {
            intersection.distance = x0;
            intersection.valid = true;
        }
        else if (x1 > epsilon) {
            intersection.distance = x1;
            intersection.valid = true;
        }

        if (intersection.valid) {
            intersection.point = r.at(intersection.distance);
            intersection.normal = normalize(intersection.point - this->position);
        }

        return intersection;
    }
};

Sphere spheres[] {
    Sphere { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left
    Sphere { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght
    Sphere { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back
    Sphere { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt
    Sphere { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm
    Sphere { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top
    Sphere { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, SPEC }, // small sphere 1
    Sphere { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, REFR }, // small sphere 2
    Sphere { 8.0f, { 50.0f, 50.0f, 50.0f }, { 13.0f, 11.5f, 11.0f }, { 1.0f, 1.0f, 1.0f }, DIFF },
//    Sphere { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

__device__
inline Intersection intersect_scene(const Ray& r, Sphere* device_spheres) {
    Intersection ret {};
    ret.distance = 1e20;

    for (int i = 0; i < 9; i++) {
        const Sphere& sphere = device_spheres[i];

        Intersection intersection = sphere.intersect(r);
        if (intersection.valid && intersection.distance < ret.distance) {
            ret = intersection;
        }
    }

    return ret;
}

// On-device RNG from https://github.com/gz/rust-raytracer

__device__
static float getrandom(uint64_t* seed0, uint64_t* seed1) {
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    union {
        float f;
        unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;

    return (res.f - 2.0f) / 2.0f;
}

__device__
float3 radiance(Ray& r, uint64_t* s1, uint64_t* s2, Sphere* device_spheres) {
    float3 color_acc = make_float3(0.0f, 0.0f, 0.0f);
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);

    for (int bounces = 0; bounces < 12; bounces++) {
        Intersection intersection = intersect_scene(r, device_spheres);
        if (!intersection.valid) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        const Sphere& obj = *intersection.object;

        float3 x = intersection.point;
        float3 n = intersection.normal;

        float3 nl = dot(n, r.direction) < 0 ? n : n * -1;
        float3 f = obj.colour;

        color_acc += mask * obj.emission;
        float3 d;

        if (obj.material == DIFF) {
            float r1 = 2 * M_PI * getrandom(s1, s2);
            float r2 = getrandom(s1, s2);
            float r2s = sqrtf(r2);

            float3 w = nl;
            float3 u = normalize(cross(fabs(w.x) > 0.1f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f), w));
            float3 v = cross(w, u);

            d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));
            x += nl * 0.03f;
            mask *= f;
        }
        else if (obj.material == SPEC) {
            d = r.direction - 2.0f * n * dot(n, r.direction);
            x += nl * 0.01f;
            mask *= f;
        }
        else if (obj.material == REFR) {
            bool into = dot(n, nl) > 0;
            float nc = 1.0f;
            float nt = 1.5f;
            float nnt = into ? nc / nt : nt / nc;
            float ddn = dot(r.direction, nl);
            float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);

            if (cos2t < 0.0f) {
                d = reflect(r.direction, n);
                x += nl * 0.01f;
            }
            else {
                float3 tdir = normalize(r.direction * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t))));

                float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
                float c = 1.0f - (into ? -ddn : dot(tdir, n));
                float Re = R0 + (1.0f - R0) * c * c * c * c * c;
                float Tr = 1.0f - Re;
                float P = 0.25f + 0.5f * Re;
                float RP = Re / P;
                float TP = Tr / (1.0f - P);

                if (getrandom(s1, s2) < 0.25f) {
                    mask *= RP;
                    d = reflect(r.direction, n);
                    // Reflection bias
                    x += nl * 0.02f;
                }
                else {
                    mask *= TP;
                    d = tdir;
                    // Transmission bias
                    x += nl * 0.000001f;
                }
            }
        }

        r.origin = x;
        r.direction = d;
    }

    return color_acc;
}

__global__
void render_kernel(float3* output, Sphere* device_spheres) {
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    uint64_t i = (height - y - 1) * width + x;

    uint64_t s1 = x;
    uint64_t s2 = y;

    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1)));

    float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f);
    float3 cy = normalize(cross(cx, cam.direction)) * .5135;
    float3 r = make_float3(0.0f);

    for (int s = 0; s < samples; s++) {
        float3 d = cam.direction + cx * ((.25 + x) / width - .5) + cy * ((.25 + y) / height - .5);

        Ray ray(cam.origin + d * 40, normalize(d));
        r += radiance(ray, &s1, &s2, device_spheres);
    }

    r /= samples;

    output[i] = clamp(r, 0.0f, 1.0f);
}

// Float to byte with gamma correction
int toInt(float n) {
    return (int)(pow(clamp(n, 0.0f, 1.0f), 1 / 2.2) * 255.0f + 0.5f);
}

void write_to_file(const char* filename, float3* buffer) {
    FILE *f = fopen(filename, "w");
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width * height; i++)  // loop over pixels, write RGB values
        fprintf(f, "%d %d %d ",
                toInt(buffer[i].x),
                toInt(buffer[i].y),
                toInt(buffer[i].z));
    fclose(f);
    printf("Saved image to '%s'\n", filename);
}

int main() {
    float3* output_h = new float3[width * height];
    float3* output_d;

    cudaMalloc(&output_d, width * height * sizeof(float3));

    Sphere* device_spheres;
    cudaMalloc(&device_spheres, sizeof(spheres));
    cudaMemcpy(device_spheres, spheres, sizeof(spheres), cudaMemcpyHostToDevice);

    dim3 block(32, 32, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    printf("Rendering...\n");

    render_kernel<<<grid, block>>>(output_d, device_spheres);

    cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(output_d);
    cudaFree(device_spheres);

    write_to_file("smallptcuda.ppm", output_h);

    delete[] output_h;
}
