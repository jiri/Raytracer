#include <stdio.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

#include "cutil_math.h"

#include "OpenImageDenoise/oidn.hpp"

#define width 1920  // screenwidth
#define height 2160 // screenheight
#define samps 32 // samples

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

//class Intersectable {
//public:
//    float3 emi;
//    float3 col;
//    Refl_t refl;
//
//    __device__
//    Intersectable(float3 emission, float3 colour, Refl_t refl, float3 pos)
//        : emi { emission }
//        , col { colour }
//        , refl { refl }
//    { }
//
//    __device__
//    virtual Intersection intersect(const Ray& ray) const = 0;
//};

struct Sphere {
    float rad;
    float3 pos;
    float3 emi;
    float3 col;
    Refl_t refl;

//    __device__
//    Sphere(float rad, float3 pos, float3 emi, float3 col, Refl_t refl)
//        // : Intersectable(emi, col, refl, pos)
//        : emi { emi }
//        , col { col }
//        , refl { refl }
//        , rad { rad }
//        , pos { pos }
//    { }

    __device__
    Intersection intersect(const Ray& r) const {
        Intersection intersection {};

        intersection.object = this;

        float3 op = pos - r.orig;    // distance from ray.orig to center sphere
        float epsilon = 0.001f;  // epsilon required to prevent floating point precision artefacts
        float b = dot(op, r.dir);    // b in quadratic equation

        float disc = b * b - dot(op, op) + rad * rad;  // discriminant quadratic equation
        if (disc < 0)
            return intersection;       // if disc < 0, no real solution (we're not interested in complex roots)
        else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant

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

// SCENE
// 9 spheres forming a Cornell box
// small enough to be in constant GPU memory
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }

// __constant__
Sphere spheres[] {
    Sphere { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left
    Sphere { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght
    Sphere { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back
    Sphere { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt
    Sphere { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm
    Sphere { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top
    Sphere { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, SPEC }, // small sphere 1
    Sphere { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, REFR }, // small sphere 2
    Sphere { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

__device__
inline Intersection intersect_scene(const Ray& r, Sphere* device_spheres) {
    Intersection ret {};
    ret.distance = 1e20;

    // printf("%f", device_spheres[0].rad);

    for (int i = 0; i < 9; i++) {
        const Sphere& sphere = device_spheres[i];

        Intersection intersection = sphere.intersect(r);
        if (intersection.valid && intersection.distance < ret.distance) {
            ret = intersection;
        }
    }

    return ret;
}

// random number generator from https://github.com/gz/rust-raytracer

__device__ static float getrandom(unsigned int *seed0, unsigned int *seed1) {
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    // Convert to float
    union {
        float f;
        unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

    return (res.f - 2.f) / 2.f;
}

// radiance function, the meat of path tracing
// solves the rendering equation:
// outgoing radiance (at a point) = emitted radiance + reflected radiance
// reflected radiance is sum (integral) of incoming radiance from all directions in hemisphere above point,
// multiplied by reflectance function of material (BRDF) and cosine incident angle
__device__ float3 radiance(Ray &r, unsigned int *s1, unsigned int *s2, Sphere* device_spheres){ // returns ray color
    float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);

    // ray bounce loop (no Russian Roulette used)
    for (int bounces = 0; bounces < 12; bounces++){  // iteration up to 4 bounces (replaces recursion in CPU code)
        // test ray for intersection with scene
        Intersection intersection = intersect_scene(r, device_spheres);
        if (!intersection.valid) {
            return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black
        }

        // else, we've got a hit!
        // compute hitpoint and normal
        const Sphere& obj = *intersection.object;  // hitobject

        float3 x = intersection.point;
        float3 n = intersection.normal;

        float3 nl = dot(n, r.dir) < 0 ? n : n * -1; // front facing normal
        float3 f = obj.col;

        // add emission of current sphere to accumulated colour
        // (first term in rendering equation sum)
        accucolor += mask * obj.emi;

        // all spheres in the scene are diffuse
        // diffuse material reflects light uniformly in all directions
        // generate new diffuse ray:
        // origin = hitpoint of previous ray in path
        // random direction in hemisphere above hitpoint (see "Realistic Ray Tracing", P. Shirley)



        // compute random ray direction on hemisphere using polar coordinates
        // cosine weighted importance sampling (favours ray directions closer to normal direction)


        // new ray origin is intersection point of previous ray with scene


        // mask *= obj.col;    // multiply with colour of object
        // mask *= dot(d,nl);  // weigh light contribution using cosine of angle between incident light and normal
        // mask *= 2;          // fudge factor

        float3 d;

        if (obj.refl == DIFF) {
            // create 2 random numbers
            float r1 = 2 * M_PI * getrandom(s1, s2); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
            float r2 = getrandom(s1, s2);  // pick random number for elevation
            float r2s = sqrtf(r2);

            // compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction
            // first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
            float3 w = nl;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w,u);

            d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));
            x += nl * 0.03f;
            mask *= f;
        }
        else if (obj.refl == SPEC) {
            // compute relfected ray direction according to Snell's law
            d = r.dir - 2.0f * n * dot(n, r.dir);

            // offset origin next path segment to prevent self intersection
            x += nl * 0.01f;

            // multiply mask with colour of object
            mask *= f;
        }
        else if (obj.refl == REFR) {
            bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
            float nc = 1.0f;  // Index of Refraction air
            float nt = 1.5f;  // Index of Refraction glass/water
            float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
            float ddn = dot(r.dir, nl);
            float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

            if (cos2t < 0.0f) { // total internal reflection
                d = reflect(r.dir, n); //d = r.dir - 2.0f * n * dot(n, r.dir);
                x += nl * 0.01f;
            }
            else { // cos2t > 0
                // compute direction of transmission ray
                float3 tdir = normalize(r.dir * nnt - n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

                float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
                float c = 1.f - (into ? -ddn : dot(tdir, n));
                float Re = R0 + (1.f - R0) * c * c * c * c * c;
                float Tr = 1 - Re; // Transmission
                float P = .25f + .5f * Re;
                float RP = Re / P;
                float TP = Tr / (1.f - P);

                // randomly choose reflection or transmission ray
                if (getrandom(s1, s2) < 0.25) { // reflection ray
                    mask *= RP;
                    d = reflect(r.dir, n);
                    x += nl * 0.02f;
                }
                else { // transmission ray
                    mask *= TP;
                    d = tdir; //r = Ray(x, tdir);
                    x += nl * 0.000001f; // epsilon must be small to avoid artefacts
                }
            }
        }

        r.orig = x; // offset ray origin slightly to prevent self intersection
        r.dir = d;
    }

    return accucolor;
}


// __global__ : executed on the device (GPU) and callable only from host (CPU)
// this kernel runs in parallel on all the CUDA threads

__global__
void render_kernel(float3* output, float3* normal, float3* albedo, Sphere* device_spheres){

    // assign a CUDA thread to every pixel (x,y)
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int i = (height - y - 1)*width + x; // index of current pixel (calculated using thread index)

    unsigned int s1 = x;  // seeds for random number generator
    unsigned int s2 = y;

// generate ray directed at lower left corner of the screen
// compute directions for all other rays by adding cx and cy increments in x and y direction
    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction)
    float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.dir)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color
    float3 n;
    float3 a;

    r = make_float3(0.0f); // reset r to zero for every pixel

    for (int s = 0; s < samps; s++){  // samples per pixel

        // compute primary ray direction
        float3 d = cam.dir + cx*((.25 + x) / width - .5) + cy*((.25 + y) / height - .5);

        // create primary ray, add incoming radiance to pixelcolor
        Ray ray(cam.orig + d * 40, normalize(d));
        r = r + radiance(ray, &s1, &s2, device_spheres)*(1. / samps);
        // n = f_normal(ray, &s1, &s2, device_spheres);
        // a = f_albedo(ray, device_spheres);
    }       // Camera rays are pushed ^^^^^ forward to start in interior

    // write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
    output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
    normal[i] = n;
    albedo[i] = a;
}

inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }  // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction

int main(){
    float3* output_h = new float3[width*height]; // pointer to memory for image on the host (system RAM)
    float3* output_d;    // pointer to memory for image on the device (GPU VRAM)

    float3* normal_h = new float3[width*height];
    float3* normal_d;

    float3* albedo_h = new float3[width*height];
    float3* albedo_d;

    // allocate memory on the CUDA device (GPU VRAM)
    cudaMalloc(&output_d, width * height * sizeof(float3));
    cudaMalloc(&normal_d, width * height * sizeof(float3));
    cudaMalloc(&albedo_d, width * height * sizeof(float3));

    Sphere* device_spheres;
    cudaError_t err = cudaMalloc(&device_spheres, sizeof(spheres));
    printf("%d", err);
    err = cudaMemcpy(device_spheres, spheres, sizeof(spheres), cudaMemcpyHostToDevice);
    printf("%d", err);

    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    printf("CUDA initialised.\nStart rendering...\n");

    // schedule threads on device and launch CUDA kernel from host
    render_kernel <<< grid, block >>>(output_d, normal_d, albedo_d, device_spheres);

    // copy results of computation from device back to host
    cudaMemcpy(output_h, output_d, width * height *sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_h, normal_d, width * height *sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(albedo_h, albedo_d, width * height *sizeof(float3), cudaMemcpyDeviceToHost);

    // free CUDA memory
    cudaFree(output_d);
    cudaFree(normal_d);
    cudaFree(albedo_d);
    cudaFree(device_spheres);

    printf("Denoising...\n");

    float3* denoised = new float3[width*height];

    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color",  output_h,  oidn::Format::Float3, width, height);
    // filter.setImage("albedo", albedo_h, oidn::Format::Float3, width, height);
    // filter.setImage("normal", normal_h, oidn::Format::Float3, width, height);
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
    delete[] normal_h;
    delete[] albedo_h;
}
