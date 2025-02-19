/**
 * @file buffer.cuh
 * @author Jasper Jeuken
 * @brief Defines a buffer for storing image data
 */
#ifndef BUFFER_H
#define BUFFER_H

#include <string>
#include <filesystem>
#include <set>
#include "scene.cuh"
#include "image.cuh"

namespace fs = std::filesystem;

/**
 * @class buffer
 * @brief Buffer for storing image data
 * 
 * This class defines a buffer for storing image data, such as color, albedo, emission, normal, depth, opacity, and denoised images.
 * It provides methods for saving the buffer to disk and normalizing the data for saving to an image file.
 */
class buffer {
public:

    /**
     * @brief Default constructor
     * 
     * @return Empty buffer
     */
    __host__ __device__ buffer() : color(nullptr), albedo(nullptr), emission(nullptr), normal(nullptr), depth(nullptr), opacity(nullptr), denoised(nullptr), d_buf(nullptr), width(0), height(0) {}

    /**
     * @brief Construct a new buffer from color, albedo, emission, normal, depth, opacity, denoised, width, and height
     * 
     * @param[in] _color Color buffer
     * @param[in] _albedo Albedo buffer
     * @param[in] _emission Emission buffer
     * @param[in] _normal Normal buffer
     * @param[in] _depth Depth buffer
     * @param[in] _opacity Opacity buffer
     * @param[in] _denoised Denoised buffer
     * @param[in] _width Image width
     * @param[in] _height Image height
     * @return Buffer that stores the image data
     */
    __host__ __device__ buffer(color* _color, color* _albedo, color* _emission, color* _normal, float* _depth, float* _opacity, color* _denoised, int _width, int _height) : color(_color), albedo(_albedo), emission(_emission), normal(_normal), depth(_depth), opacity(_opacity), denoised(_denoised), d_buf(nullptr), width(_width), height(_height) {}

    /**
     * @brief Destructor
     * 
     * Frees the color, albedo, emission, normal, depth, opacity, denoised, and device buffers.
     */
    __host__ ~buffer() {
        checkCudaErrors(cudaDeviceSynchronize());

        // Free the color buffer
        if (color) {
            free_object(color);
            color = nullptr;
        }

        // Free the albedo buffer
        if (albedo) {
            free_object(albedo);
            albedo = nullptr;
        }

        // Free the emission buffer
        if (emission) {
            free_object(emission);
            emission = nullptr;
        }

        // Free the normal buffer
        if (normal) {
            free_object(normal);
            normal = nullptr;
        }

        // Free the depth buffer
        if (depth) {
            free_object(depth);
            depth = nullptr;
        }

        // Free the opacity buffer
        if (opacity) {
            free_object(opacity);
            opacity = nullptr;
        }

        // Free the device buffer
        if (d_buf) {
            free_object(d_buf);
            d_buf = nullptr;
        }
    }

    /**
     * @brief Select a pixel from the color buffer
     * 
     * @param[in] i Pixel index
     * @return Selected pixel
     */
    __host__ __device__ vec3& operator[](int i) { return color[i]; }

    /**
     * @brief Select a pixel from the color buffer
     * 
     * @param[in] i Pixel index
     * @return Selected pixel
     */
    __host__ __device__ const vec3& operator[](int i) const { return color[i]; }

    /**
     * @brief Redirect the pointer to the color buffer
     * 
     * @return Pointer to the color buffer
     */
    __host__ __device__ vec3* operator->() { return color; }

    /**
     * @brief Redirect the pointer to the color buffer
     * 
     * @return Pointer to the color buffer 
     */
    __host__ __device__ const vec3* operator->() const { return color; }

    /**
     * @brief Get the size of the buffer
     * 
     * @return Size of the buffer
     */
    __host__ __device__ int size() const { return width * height; }

    /**
     * @brief Get the device buffer object
     * 
     * @return Device buffer object
     */
    __host__ buffer* get_device_buffer() {
        return d_buf;
    }

    /**
     * @brief Save all passes to disk
     * 
     * @param[in] folder_path Path to the folder where the passes will be saved
     * @param[in] format Image format (e.g., `png`, `jpg`, `bmp`)
     */
    void save_all_passes(std::string folder_path, std::string format) {
        fs::create_directories(folder_path);
        save_passes(folder_path, format, {"color", "albedo", "emission", "normal", "depth", "opacity", "denoised"});
    }

    /**
     * @brief Save selected passes to disk
     * 
     * @param[in] folder_path Path to the folder where the passes will be saved 
     * @param[in] format Image format (e.g., `png`, `jpg`, `bmp`)
     * @param[in] passes Set of passes to save (e.g., `{"color", "albedo", "emission", "normal", "depth", "opacity", "denoised"}`)
     */
    void save_passes(std::string folder_path, std::string format, std::set<std::string> passes) {
        std::cout << "Saving render passes (";
        for (auto pass : passes) {
            std::cout << " " << pass << " ";
        }
        std::cout << ")...\n";

        fs::create_directories(folder_path);
        if (passes.count("color")) {
            write_image((fs::u8path(folder_path) / fs::u8path("color." + format)).string(), normalize_color(), width, height, 3, format);
        }
        if (passes.count("albedo")) {
            write_image((fs::u8path(folder_path) / fs::u8path("albedo." + format)).string(), normalize_albedo(), width, height, 3, format);
        }
        if (passes.count("emission")) {
            write_image((fs::u8path(folder_path) / fs::u8path("emission." + format)).string(), normalize_emission(), width, height, 3, format);
        }
        if (passes.count("normal")) {
            write_image((fs::u8path(folder_path) / fs::u8path("normal." + format)).string(), normalize_normal(), width, height, 3, format);
        }
        if (passes.count("depth")) {
            write_image((fs::u8path(folder_path) / fs::u8path("depth." + format)).string(), normalize_depth(), width, height, 3, format);
        }
        if (passes.count("opacity")) {
            write_image((fs::u8path(folder_path) / fs::u8path("opacity." + format)).string(), normalize_opacity(), width, height, 3, format);
        }
        if (passes.count("denoised")) {
            write_image((fs::u8path(folder_path) / fs::u8path("denoised." + format)).string(), normalize_denoised(), width, height, 3, format);
        }
        std::cout << " - Images saved.\n";
    }

    /**
     * @brief Normalize the color buffer
     * 
     * @return Normalized color buffer
     */
    unsigned char* normalize_color() {
        unsigned char* data = new unsigned char[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            data[i * 3 + 0] = color_value(gamma_correct(color[i].x()));
            data[i * 3 + 1] = color_value(gamma_correct(color[i].y()));
            data[i * 3 + 2] = color_value(gamma_correct(color[i].z()));
        }
        return data;
    }

    /**
     * @brief Normalize the albedo buffer
     * 
     * @return Normalized albedo buffer
     */
    unsigned char* normalize_albedo() {
        unsigned char* data = new unsigned char[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            data[i * 3 + 0] = color_value(gamma_correct(albedo[i].x()));
            data[i * 3 + 1] = color_value(gamma_correct(albedo[i].y()));
            data[i * 3 + 2] = color_value(gamma_correct(albedo[i].z()));
        }
        return data;
    }

    /**
     * @brief Normalize the emission buffer
     * 
     * @return Normalized emission buffer
     */
    unsigned char* normalize_emission() {
        unsigned char* data = new unsigned char[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            data[i * 3 + 0] = color_value(gamma_correct(emission[i].x()));
            data[i * 3 + 1] = color_value(gamma_correct(emission[i].y()));
            data[i * 3 + 2] = color_value(gamma_correct(emission[i].z()));
        }
        return data;
    }

    /**
     * @brief Normalize the normal buffer
     * 
     * @return Normalized normal buffer
     */
    unsigned char* normalize_normal() {
        unsigned char* data = new unsigned char[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            data[i * 3 + 0] = color_value((normal[i].x() + 1.0f) / 2.0f);
            data[i * 3 + 1] = color_value((normal[i].y() + 1.0f) / 2.0f);
            data[i * 3 + 2] = color_value((normal[i].z() + 1.0f) / 2.0f);
        }
        return data;
    }

    /**
     * @brief Normalize the depth buffer
     * 
     * @return Normalized depth buffer
     */
    unsigned char* normalize_depth() {
        unsigned char* data = new unsigned char[width * height * 3];
        float max_depth = 0.0f;
        float min_depth = std::numeric_limits<float>::max();
        for (int i = 0; i < width * height; i++) {
            if (!isnan(depth[i])) {
                max_depth = maxf(max_depth, depth[i]);
                min_depth = minf(min_depth, depth[i]);
            }
        }
        for (int i = 0; i < width * height; i++) {
            data[i * 3 + 0] = color_value(map(minf(depth[i], max_depth), min_depth, max_depth, 0.0f, 1.0f));
            data[i * 3 + 1] = color_value(map(minf(depth[i], max_depth), min_depth, max_depth, 0.0f, 1.0f));
            data[i * 3 + 2] = color_value(map(minf(depth[i], max_depth), min_depth, max_depth, 0.0f, 1.0f));
        }
        return data;
    }

    /**
     * @brief Normalize the opacity buffer
     * 
     * @return Normalized opacity buffer
     */
    unsigned char* normalize_opacity() {
        unsigned char* data = new unsigned char[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            data[i * 3 + 0] = color_value(opacity[i]);
            data[i * 3 + 1] = color_value(opacity[i]);
            data[i * 3 + 2] = color_value(opacity[i]);
        }
        return data;
    }

    /**
     * @brief Normalize the denoised buffer
     * 
     * @return Normalized denoised buffer
     */
    unsigned char* normalize_denoised() {
        unsigned char* data = new unsigned char[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            data[i * 3 + 0] = color_value(gamma_correct(denoised[i].x()));
            data[i * 3 + 1] = color_value(gamma_correct(denoised[i].y()));
            data[i * 3 + 2] = color_value(gamma_correct(denoised[i].z()));
        }
        return data;
    }

    /**
     * @brief Apply gamma correction to a value
     * 
     * @param[in] value Value to correct
     * @return Gamma-corrected value
     */
    float gamma_correct(float value) {
        return std::pow(value, 1.0f / gamma);
    }


public:
    vec3* color; ///< Color buffer
    vec3* albedo; ///< Albedo buffer
    vec3* emission; ///< Emission buffer
    vec3* normal; ///< Normal buffer
    float* depth; ///< Depth buffer
    float* opacity; ///< Opacity buffer
    vec3* denoised; ///< Denoised buffer
    int width; ///< Image width
    int height; ///< Image height
    int samples_per_pixel; ///< Samples per pixel
    int max_bounces; ///< Maximum number of bounces
    float gamma; ///< Gamma correction factor
    buffer* d_buf; ///< Device buffer
};


/**
 * @brief Create a buffer object from a scene
 * 
 * @param[in] sc Scene object
 * @return New buffer object
 */
std::unique_ptr<buffer> create_buffer(std::unique_ptr<scene>& sc) {
    buffer* buf = new buffer();
    int buf_size = sc->render_info.image_width * sc->render_info.image_height;

    vec3* color_data;
    checkCudaErrors(cudaMallocManaged(&color_data, buf_size * sizeof(vec3)));

    vec3* albedo_data;
    checkCudaErrors(cudaMallocManaged(&albedo_data, buf_size * sizeof(vec3)));

    vec3* emission_data;
    checkCudaErrors(cudaMallocManaged(&emission_data, buf_size * sizeof(vec3)));

    vec3* normal_data;
    checkCudaErrors(cudaMallocManaged(&normal_data, buf_size * sizeof(vec3)));

    float* depth_data;
    checkCudaErrors(cudaMallocManaged(&depth_data, buf_size * sizeof(float)));

    float* opacity_data;
    checkCudaErrors(cudaMallocManaged(&opacity_data, buf_size * sizeof(float)));

    buf->color = color_data;
    buf->albedo = albedo_data;
    buf->emission = emission_data;
    buf->normal = normal_data;
    buf->depth = depth_data;
    buf->opacity = opacity_data;
    buf->width = sc->render_info.image_width;
    buf->height = sc->render_info.image_height;
    buf->d_buf = copy_object(*buf);
    buf->samples_per_pixel = sc->render_info.samples_per_pixel;
    buf->max_bounces = sc->render_info.max_bounces;
    buf->gamma = sc->render_info.gamma;
    
    return std::unique_ptr<buffer>(buf);
}


#endif