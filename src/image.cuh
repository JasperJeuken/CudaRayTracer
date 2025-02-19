/**
 * @file image.cuh
 * @author Jasper Jeuken
 * @brief Defines functions to write images to disk
 */
#ifndef IMAGE_H
#define IMAGE_H

#pragma nv_diag_suppress = 550  // Suppress STB unused variable warnings

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include <string>
#include <stb_image_write.h>

/**
 * @brief Write a buffer to a PNG file
 * 
 * @param[in] filename Filename to write to
 * @param[in] data Buffer to write
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] channels Image channels
 * @param[in] bytes_per_pixel Image bytes per pixel
 */
void write_png(std::string filename, unsigned char* data, int width, int height, int channels, int bytes_per_pixel = 3) {
    stbi_write_png(filename.c_str(), width, height, channels, data, bytes_per_pixel * width);
}

/**
 * @brief Write a buffer to a JPG file
 * 
 * @param[in] filename Filename to write to
 * @param[in] data Buffer to write
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] channels Image channels
 * @param[in] bytes_per_pixel Image bytes per pixel
 * @param[in] quality Image quality
 */
void write_jpg(std::string filename, unsigned char* data, int width, int height, int channels, int bytes_per_pixel = 3, int quality = 100) {
    stbi_write_jpg(filename.c_str(), width, height, channels, data, quality);
}

/**
 * @brief Write a buffer to a BMP file
 * 
 * @param[in] filename Filename to write to
 * @param[in] data Buffer to write
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] channels Image channels
 * @param[in] bytes_per_pixel Image bytes per pixel
 */
void write_bmp(std::string filename, unsigned char* data, int width, int height, int channels, int bytes_per_pixel = 3) {
    stbi_write_bmp(filename.c_str(), width, height, channels, data);
}

/**
 * @brief Write a buffer to a TGA file
 * 
 * @param[in] filename Filename to write to
 * @param[in] data Buffer to write
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] channels Image channels
 * @param[in] bytes_per_pixel Image bytes per pixel
 */
void write_tga(std::string filename, unsigned char* data, int width, int height, int channels, int bytes_per_pixel = 3) {
    stbi_write_tga(filename.c_str(), width, height, channels, data);
}

/**
 * @brief Write a buffer to a HDR file
 * 
 * @param[in] filename Filename to write to
 * @param[in] data Buffer to write
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] channels Image channels
 * @param[in] bytes_per_pixel Image bytes per pixel
 */
void write_hdr(std::string filename, float* data, int width, int height, int channels, int bytes_per_pixel = 3) {
    stbi_write_hdr(filename.c_str(), width, height, channels, data);
}

/**
 * @brief Write a buffer to an image file
 * 
 * @param[in] filename Filename to write to
 * @param[in] data Buffer to write
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] channels Image channels
 * @param[in] format Image format (`png`, `jpg`, `bmp`, `tga`, `hdr`)
 * @param[in] bytes_per_pixel Image bytes per pixel
 * @param[in] quality Image quality
 */
void write_image(std::string filename, unsigned char* data, int width, int height, int channels, std::string format, int bytes_per_pixel = 3, int quality = 100) {
    if (format == "png") {
        write_png(filename, data, width, height, channels, bytes_per_pixel);
    } else if (format == "jpg") {
        write_jpg(filename, data, width, height, channels, bytes_per_pixel, quality);
    } else if (format == "bmp") {
        write_bmp(filename, data, width, height, channels, bytes_per_pixel);
    } else if (format == "tga") {
        write_tga(filename, data, width, height, channels, bytes_per_pixel);
    } else if (format == "hdr") {
        std::vector<float> float_data(width * height * channels);
        for (int i = 0; i < width * height * channels; i++) {
            float_data[i] = static_cast<float>(data[i]) / 255.0f;
        }
        write_hdr(filename, float_data.data(), width, height, channels, bytes_per_pixel);
    } else {
        error_with_message("Unsupported image format: " + format);
    }
}

#endif