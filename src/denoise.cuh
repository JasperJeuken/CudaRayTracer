/**
 * @file denoise.cuh
 * @author Jasper Jeuken
 * @brief Defines a denoising function
 */
#ifndef DENOISE_H
#define DENOISE_H

#include "buffer.cuh"
#include <OpenImageDenoise/oidn.h>


/**
 * @brief Denoises a buffer
 * 
 * @param[in] buf Buffer to denoise
 * @see https://www.openimagedenoise.org
 * 
 * Uses the Open Image Denoise library to denoise a buffer. The buffer must have a color pass rendered.
 * If the buffer has an albedo or normal pass, these will be used as additional inputs.
 */
void denoise(std::unique_ptr<buffer>& buf) {
    std::cout << "Denoising image...\n";
    if (!buf->color) {
        error_with_message("Denoise failed, color pass not rendered");
    }

    // Create OIDN device
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
    oidnCommitDevice(device);

    // Create OIDN buffers
    size_t buf_size = buf->width * buf->height * 3;
    OIDNBuffer color_buffer = oidnNewBuffer(device, buf_size * sizeof(float));
    OIDNBuffer albedo_buffer = nullptr;
    OIDNBuffer normal_buffer = nullptr;
    OIDNBuffer output_buffer = oidnNewBuffer(device, buf_size * sizeof(float));

    // Copy data into OIDN buffers
    oidnWriteBufferAsync(color_buffer, 0, buf_size * sizeof(float), reinterpret_cast<float*>(buf->color));
    if (buf->albedo) {
        albedo_buffer = oidnNewBuffer(device, buf_size * sizeof(float));
        oidnWriteBufferAsync(albedo_buffer, 0, buf_size * sizeof(float), reinterpret_cast<float*>(buf->albedo));
    }
    if (buf->normal) {
        normal_buffer = oidnNewBuffer(device, buf_size * sizeof(float));
        oidnWriteBufferAsync(normal_buffer, 0, buf_size * sizeof(float), reinterpret_cast<float*>(buf->normal));
    }
    oidnSyncDevice(device);
    
    // Create filter
    OIDNFilter filter = oidnNewFilter(device, "RT");
    oidnSetFilterImage(filter, "color", color_buffer, OIDN_FORMAT_FLOAT3, buf->width, buf->height, 0, 0, 0);
    if (buf->albedo) oidnSetFilterImage(filter, "albedo", albedo_buffer, OIDN_FORMAT_FLOAT3, buf->width, buf->height, 0, 0, 0);
    if (buf->normal) oidnSetFilterImage(filter, "normal", normal_buffer, OIDN_FORMAT_FLOAT3, buf->width, buf->height, 0, 0, 0);
    oidnSetFilterImage(filter, "output", output_buffer, OIDN_FORMAT_FLOAT3, buf->width, buf->height, 0, 0, 0);
    oidnCommitFilter(filter);

    // Run filter and check for errors
    oidnExecuteFilter(filter);
    std::cerr << " - Started OIDN.\n";
    const char* error;
    if (oidnGetDeviceError(device, &error) != OIDN_ERROR_NONE) {
        std::cerr << "Denoise failed: " << error << "\n";
        oidnReleaseFilter(filter);
        oidnReleaseDevice(device);
        return;
    }

    // Read output
    buf->denoised = new vec3[buf->width * buf->height];
    oidnReadBuffer(output_buffer, 0, buf_size * sizeof(float), reinterpret_cast<float*>(buf->denoised));

    // Cleanup
    oidnReleaseFilter(filter);
    oidnReleaseBuffer(color_buffer);
    if (buf->albedo) oidnReleaseBuffer(albedo_buffer);
    if (buf->normal) oidnReleaseBuffer(normal_buffer);
    oidnReleaseBuffer(output_buffer);
    oidnReleaseDevice(device);
    std::cerr << " - Finished denoise.\n";
}

#endif