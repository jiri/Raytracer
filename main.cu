#include <stdio.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

#include "cutil_math.h"

#include "OpenImageDenoise/oidn.hpp"

#define width 1920  // screenwidth
#define height 2160 // screenheight
#define samps 128 // samples

struct Ray {
    float3 orig; // ray origin
    float3 dir;  // ray direction

    float3& origin;
    float3& direction;

    __host__ __device__
    Ray(float3 o_, float3 d_)
        : orig(o_)
        , dir(d_)
        , origin { this->orig }
        , direction { this->dir }
    { }
};

enum Refl_t {
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
    float rad;
    float3 pos;
    float3 emi;
    float3 col;
    Refl_t refl;

    __device__
    Intersection intersect(const Ray& r) const {
        Intersection intersection {};

        intersection.object = this;

        float3 op = pos - r.orig;
        float epsilon = 0.001f;
        float b = dot(op, r.dir);

        float disc = b * b - dot(op, op) + rad * rad;
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
            intersection.point = r.orig + r.dir * intersection.distance;
            intersection.normal = normalize(intersection.point - this->pos);
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
static float getrandom(unsigned int* seed0, unsigned int* seed1) {
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
float3 radiance(Ray& r, unsigned int* s1, unsigned int* s2, Sphere* device_spheres) {
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

        float3 nl = dot(n, r.dir) < 0 ? n : n * -1;
        float3 f = obj.col;

        color_acc += mask * obj.emi;
        float3 d;

        if (obj.refl == DIFF) {
            float r1 = 2 * M_PI * getrandom(s1, s2);
            float r2 = getrandom(s1, s2);
            float r2s = sqrtf(r2);

            float3 w = nl;
            float3 u = normalize(cross((fabs(w.x) > 0.1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);

            d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));
            x += nl * 0.03f;
            mask *= f;
        }
        else if (obj.refl == SPEC) {
            d = r.dir - 2.0f * n * dot(n, r.dir);
            x += nl * 0.01f;
            mask *= f;
        }
        else if (obj.refl == REFR) {
            bool into = dot(n, nl) > 0;
            float nc = 1.0f;
            float nt = 1.5f;
            float nnt = into ? nc / nt : nt / nc;
            float ddn = dot(r.dir, nl);
            float cos2t = 1.0f - nnt * nnt * (1.f - ddn * ddn);

            if (cos2t < 0.0f) {
                d = reflect(r.dir, n);
                x += nl * 0.01f;
            }
            else {
                float3 tdir = normalize(r.dir * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t))));

                float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
                float c = 1.f - (into ? -ddn : dot(tdir, n));
                float Re = R0 + (1.f - R0) * c * c * c * c * c;
                float Tr = 1 - Re;
                float P = .25f + .5f * Re;
                float RP = Re / P;
                float TP = Tr / (1.f - P);

                if (getrandom(s1, s2) < 0.25) {
                    mask *= RP;
                    d = reflect(r.dir, n);
                    x += nl * 0.02f;
                }
                else {
                    mask *= TP;
                    d = tdir;
                    x += nl * 0.000001f;
                }
            }
        }

        r.orig = x;
        r.dir = d;
    }

    return color_acc;
}

__global__
void render_kernel(float3* output, Sphere* device_spheres) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int i = (height - y - 1) * width + x;

    unsigned int s1 = x;
    unsigned int s2 = y;

    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1)));

    float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f);
    float3 cy = normalize(cross(cx, cam.dir)) * .5135;
    float3 r = make_float3(0.0f);


    for (int s = 0; s < samps; s++) {
        float3 d = cam.dir + cx * ((.25 + x) / width - .5) + cy * ((.25 + y) / height - .5);

        Ray ray(cam.orig + d * 40, normalize(d));
        r = r + radiance(ray, &s1, &s2, device_spheres)*(1. / samps);
    }

    output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

int main(){
    float3* output_h = new float3[width * height];
    float3* output_d;

    cudaMalloc(&output_d, width * height * sizeof(float3));

    Sphere* device_spheres;
    cudaMalloc(&device_spheres, sizeof(spheres));
    cudaMemcpy(device_spheres, spheres, sizeof(spheres), cudaMemcpyHostToDevice);

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    printf("Rendering...\n");

    render_kernel<<<grid, block>>>(output_d, device_spheres);

    cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(output_d);
    cudaFree(device_spheres);

    printf("Denoising...\n");

    float3* denoised = new float3[width*height];

    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color",  output_h,  oidn::Format::Float3, width, height);
    filter.setImage("output", denoised, oidn::Format::Float3, width, height);
    filter.commit();

    filter.execute();

    printf("Done!\n");

    FILE *f = fopen("smallptcuda.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width*height; i++)  // loop over pixels, write RGB values
        fprintf(f, "%d %d %d ",
                toInt(denoised[i].x),
                toInt(denoised[i].y),
                toInt(denoised[i].z));

    printf("Saved image to 'smallptcuda.ppm'\n");

    delete[] output_h;
}
