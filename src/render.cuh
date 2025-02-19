/**
 * @file render.cuh
 * @author Jasper Jeuken
 * @brief Defines the rendering kernel
 */
#ifndef RENDER_H
#define RENDER_H

#include "buffer.cuh"
#include "interval.cuh"
#include "scene.cuh"

#include <time.h>

/**
 * @struct ray_info
 * @brief Information about a ray
 */
struct ray_info {
    color col; ///< Ray color
    color albedo; ///< Ray albedo
    color emission; ///< Ray emission
    vec3 normal; ///< Ray normal
    float depth; ///< Ray depth
    float opacity; ///< Ray opacity
};

/**
 * @brief Determine the information of a ray
 * 
 * @param[in] r Ray to determine information for
 * @param[in] sc Scene to render
 * @param[in] rand_state Random state
 * @param[out] result Output ray information
 */
__device__ void get_ray_info(const ray& r, device_scene* sc, curandState* rand_state, ray_info& result) {
    ray cur_ray = r;
    color cur_attenuation(1, 1, 1);
    hit_record rec;

    // Keep doing bounces until the maximum number of bounces is reached
    int bounces = 0;
    int max_bounces = sc->d_camera->render_info.max_bounces;
    while (bounces < max_bounces) { 

        // Check if the ray hits an object
        if (bvh4_hit(cur_ray, interval(0.001f, d_infinity), rec, rand_state, sc->d_bvh4, sc->d_objects, sc->d_textures)) {
            ray scattered;
            color attenuation;
            
            // Check if the material scatters the ray
            if (sc->d_materials[rec.mat_idx]->scatter(cur_ray, rec, attenuation, scattered, rand_state, sc->d_textures)) {
                if (bounces == 0) {
                    result.albedo = attenuation;
                    result.normal = vec3::unit_vector(rec.normal);
                    result.depth = rec.t * cur_ray.direction().length();
                    result.opacity = 1.0f;
                }
                cur_attenuation *= attenuation;
                cur_ray = scattered;
                bounces++;

            // If the material does not scatter the ray, emit light
            } else {
                color emitted = sc->d_materials[rec.mat_idx]->emit(rec.u, rec.v, rec.p, sc->d_textures);
                result.col = cur_attenuation * emitted;
                if (bounces == 0) {
                    result.albedo = emitted;
                    result.emission = emitted;
                    result.normal = vec3::unit_vector(rec.normal);
                    result.depth = rec.t * cur_ray.direction().length();;
                    result.opacity = 1.0f;
                }
                return;
            }
        } else {
            vec3 p = vec3::unit_vector(cur_ray.direction());
            float theta = acos(-p.y());
            float phi = atan2(-p.z(), p.x()) + d_pi;
            float u = phi / (2 * d_pi);
            float v = theta / d_pi;
            vec3 c = sc->d_textures[sc->background_index]->value(u, v, p, sc->d_textures);
            result.col = cur_attenuation * c;
            if (bounces == 0) {
                result.albedo = result.col;
                result.emission = color(0, 0, 0);
                result.normal = vec3(0.0f, 0.0f, 0.0f);
                result.depth = nanf("");
                result.opacity = 0.0f;
            }
            return;
        }
    }

    // If the ray has bounced too many times, return black
    result.col = color(0, 0, 0);
    result.albedo = color(0, 0, 0);
}

/**
 * @brief Kernel to render a scene
 * 
 * @param[in] sc Scene to render
 * @param[in] buf Buffer to render to
 * @param[in] samples Number of samples to render
 * @param[in] samples_done Number of samples already rendered
 */
__global__ void render_kernel(device_scene* sc, buffer* buf, int samples = -1, int samples_done = 0) {
    // Determine pixel index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sc->d_camera->render_info.image_width || j >= sc->d_camera->render_info.image_height) return;
    int pixel_index = j * sc->d_camera->render_info.image_width + i;

    // Initialize random state
    curandState rand_state;
    curand_init(1984 + pixel_index, samples_done, 0, &rand_state);

    // Determine how many samples to run
    if (samples < 0) {
        samples = sc->d_camera->render_info.samples_per_pixel;
    } else {
        samples = minf(samples, sc->d_camera->render_info.samples_per_pixel);
    }

    // Calculate pixel information
    for (int s = 0; s < samples; s++) {
        int current_sample = samples_done + s + 1;
        
        // Get ray and calculate information
        ray_info result;
        ray r = sc->d_camera->get_ray(i, j, &rand_state);
        get_ray_info(r, sc, &rand_state, result);

        // Write first sample
        if (current_sample == 1) {
            buf->color[pixel_index] = result.col;
            buf->albedo[pixel_index] = result.albedo;
            buf->emission[pixel_index] = result.emission;
            buf->normal[pixel_index] = result.normal;
            buf->depth[pixel_index] = result.depth;
            buf->opacity[pixel_index] = result.opacity;
            continue;
        }

        // Write other samples
        buf->color[pixel_index] = (buf->color[pixel_index] * (current_sample - 1) + result.col) / current_sample;
        buf->albedo[pixel_index] = (buf->albedo[pixel_index] * (current_sample - 1) + result.albedo) / current_sample;
        buf->emission[pixel_index] = (buf->emission[pixel_index] * (current_sample - 1) + result.emission) / current_sample;
        buf->normal[pixel_index] = (buf->normal[pixel_index] * (current_sample - 1) + result.normal) / current_sample;
        buf->depth[pixel_index] = (buf->depth[pixel_index] * (current_sample - 1) + result.depth) / current_sample;
        buf->opacity[pixel_index] = (buf->opacity[pixel_index] * (current_sample - 1) + result.opacity) / current_sample;
    }
}

/**
 * @brief Render a scene
 * 
 * @param[in] sc Scene to render
 * @param[in] buf Buffer to render to
 * @param[in] tx Number of threads in x direction
 * @param[in] ty Number of threads in y direction
 * @param[in] samples Number of samples to render
 * @param[in] samples_done Number of samples already rendered
 * @return Render duration (seconds)
 */
__host__ float render(std::unique_ptr<scene>& sc, std::unique_ptr<buffer>& buf, int tx = 8, int ty = 8, int samples = -1, int samples_done = 0) {
    // Calculate number of blocks and threads per block
    dim3 blocks(sc->render_info.image_width / tx + 1, sc->render_info.image_height / ty + 1);
    dim3 threads(tx, ty);

    // Run render kernel
    clock_t start, stop;
    start = clock();
    render_kernel<<<blocks, threads>>>(sc->get_device_scene(), buf->get_device_buffer(), samples, samples_done);
    if (samples_done == 0) std::cout << " - Started render kernel.\n";
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();

    return (float)(stop - start) / CLOCKS_PER_SEC;
}

#endif