/**
 * @file texture.cuh
 * @author Jasper Jeuken
 * @brief Defines texture classes
 */
#ifndef TEXTURE_H
#define TEXTURE_H

#pragma nv_diag_suppress = 550  // Suppress STB unused variable warnings

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "image.cuh"
#include "vec3.cuh"
#include <stb_image.h>
#include <cuda_runtime.h>

/// @brief Enum for texture types
enum class texture_type {
    SOLID_COLOR,
    CHECKER,
    IMAGE
};

/**
 * @class texture
 * @brief Base texture
 */
class _texture {
public:
    texture_type type; ///< Texture type

    /**
     * @brief Default constructor
     * 
     * @return Empty texture
     */
    __host__ _texture() : type(texture_type::SOLID_COLOR) {}

    /**
     * @brief Construct a new base texture object
     * 
     * @param[in] _type Texture type
     * @return Constructed texture
     */
    __host__ _texture(texture_type _type) : type(_type) {}

    /// @brief Virtual destructor
    __host__ virtual ~_texture() {}

    /**
     * @brief Determine the value of the texture at a given u-v coordinate
     * 
     * @param[in] u Texture coordinate u
     * @param[in] v Texture coordinate v
     * @param[in] p Hit point
     * @param[in] textures Textures in the scene
     * @return Color at specified point
     */
    __device__ color value(float u, float v, const point3& p, _texture** textures) const;
};

/**
 * @struct solid_color
 * @brief Solid color texture
 */
struct solid_color : public _texture {
    color albedo; ///< Texture albedo

    /**
     * @brief Default constructor
     * 
     * @return Black solid color texture
     */
    __host__ solid_color() : albedo(0, 0, 0), _texture(texture_type::SOLID_COLOR) {}

    /**
     * @brief Construct a new solid color object
     * 
     * @param[in] _albedo Color
     * @return Constructed solid color texture
     */
    __host__ solid_color(const color& _albedo) : albedo(_albedo), _texture(texture_type::SOLID_COLOR) {}
};

/**
 * @struct checker
 * @brief Checker texture
 */
struct checker : public _texture {
    int odd_idx; ///< Index of odd texture
    int even_idx; ///< Index of even texture
    float inv_scale; ///< Inverse of checker scale

    /**
     * @brief Default constructor
     * 
     * @return Empty texture
     */
    __host__ checker() :  odd_idx(-1), even_idx(-1), inv_scale(-1.0f), _texture(texture_type::CHECKER) {}

    /**
     * @brief Construct a new checker texture
     * 
     * @param[in] _odd_idx Index of odd texture
     * @param[in] _even_idx Index of even texture
     * @param[in] _scale Scale of the checker pattern
     * @return Constructed checker texture
     */
    __host__ checker(int _odd_idx, int _even_idx, float _scale = 1.0f) : odd_idx(_odd_idx), even_idx(_even_idx), inv_scale(1.0f / _scale), _texture(texture_type::CHECKER) {}
};

/**
 * @struct image
 * @brief Image texture
 */
struct image: public _texture {
    unsigned char* data; ///< Raw image data
    cudaTextureObject_t tex; ///< CUDA texture object
    unsigned char* d_pitched_data; ///< Device pitched image data
    int width; ///< Image width
    int height; ///< Image height
    int channels; ///< Number of image channels

    /**
     * @brief Default constructor
     * 
     * @return Empty texture
     */
    __host__ image() : data(nullptr), tex(0), width(0), height(0), channels(0), _texture(texture_type::IMAGE) {}

    /**
     * @brief Construct a new image texture from an image file
     * 
     * @param[in] filename Name of image to load
     * @param[in] hdr_gamma Gamma adjustment factor (HDR only)
     * @param[in] hdr_scale Scale adjustment factor (HDR only)
     * @param[in] desired_channels Desired number of channels (0=auto)
     * @param[in] flip_y whether to flip image vertically
     * @return Constructed image
     */
    __host__ image(std::string filename, float hdr_gamma = 2.2f, float hdr_scale = 1.0f, int desired_channels = 0, bool flip_y = false) : _texture(texture_type::IMAGE) {
        stbi_hdr_to_ldr_gamma(hdr_gamma);
        stbi_hdr_to_ldr_scale(hdr_scale);
        stbi_set_flip_vertically_on_load(flip_y);
        data = stbi_load(filename.c_str(), &width, &height, &channels, desired_channels);

        // CUDA only supports 1 or 4 channel texture
        if (channels == 3) {
            add_alpha_channel();
        } else if (channels != 1 && channels != 4) {
            error_with_message("Images with 2 channels not supported");
        }

        // Create CUDA 2D texture
        create_texture();
    }

    /**
     * @brief Add an alpha channel to the image (full 255)
     */
    __host__ void add_alpha_channel() {
        // Allocate new array
        unsigned char* new_data = new unsigned char[width * height * 4];

        // Copy existing data into new data
        for (int i = 0; i < width * height; i++) {
            int rgb_index = i * 3;
            int rgba_index = i * 4;

            // Copy RGB values
            new_data[rgba_index + 0] = data[rgb_index + 0];
            new_data[rgba_index + 1] = data[rgb_index + 1];
            new_data[rgba_index + 2] = data[rgb_index + 2];
            new_data[rgba_index + 3] = 255;
        }
        
        // Replace data
        delete[] data;
        data = new_data;
        channels = 4;
    }

    /**
     * @brief Create a CUDA texture from the image
     */
    __host__ void create_texture() {
        // Copy image data to pitched memory
        size_t pitch;
        checkCudaErrors(cudaMallocPitch(&d_pitched_data, &pitch, width * channels, height));
        checkCudaErrors(cudaMemcpy2D(d_pitched_data, pitch, data, width * channels, width * channels, height, cudaMemcpyHostToDevice));

        // Create resource descriptor
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = d_pitched_data;
        res_desc.res.pitch2D.pitchInBytes = pitch;
        res_desc.res.pitch2D.width = width;
        res_desc.res.pitch2D.height = height;

        // Set channel descriptor
        if (channels == 1) {
            res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar1>();
        } else if (channels == 4) {
            res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
        } else {
            error_with_message("Unsupported number of image channels");
        }

        // Create texture descriptor
        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;

        // Create texture object
        checkCudaErrors(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));
    }

    /**
     * @brief Free the image texture and associated data from the host and device memory
     */
    __host__ ~image() {
        if (d_pitched_data) {
            checkCudaErrors(cudaFree(d_pitched_data));
            d_pitched_data = nullptr;
        }
        if (tex) {
            checkCudaErrors(cudaDestroyTextureObject(tex));
            tex = 0;
        }
        if (data) {
            delete[] data;
            data = nullptr;
        }
    }
};

/**
 * @brief Get the solid color value at a specified point
 * 
 * @param[in] u Texture coordinate u
 * @param[in] v Texture coordinate v
 * @param[in] p Hit point
 * @param[in] textures Textures in scene
 * @param[in] object_data Solid color data
 * @return Color of texture 
 */
__device__ color value_solid_color(float u, float v, const point3& p, _texture** textures, const void* object_data) {
    solid_color* sc = reinterpret_cast<solid_color*>(const_cast<void*>(object_data));
    return sc->albedo;
}
\
/**
 * @brief Get the checker color value at a specified point
 * 
 * @param[in] u Texture coordinate u
 * @param[in] v Texture coordinate v
 * @param[in] p Hit point
 * @param[in] textures Textures in scene
 * @param[in] object_data Checker data
 * @return Color of texture 
 */
__device__ color value_checker(float u, float v, const point3& p, _texture** textures, const void* object_data) {
    checker* c = reinterpret_cast<checker*>(const_cast<void*>(object_data));
    int x_int = int(floor(c->inv_scale * p.x()));
    int y_int = int(floor(c->inv_scale * p.y()));
    int z_int = int(floor(c->inv_scale * p.z()));
    bool is_even = (x_int + y_int + z_int) % 2 == 0;
    return is_even ? textures[c->even_idx]->value(u, v, p, textures) : textures[c->odd_idx]->value(u, v, p, textures);
}

/**
 * @brief Get the image color value at a specified point
 * 
 * @param[in] u Texture coordinate u
 * @param[in] v Texture coordinate v
 * @param[in] p Hit point
 * @param[in] textures Textures in scene
 * @param[in] object_data Image data
 * @return Color of texture 
 */
__device__ color value_image(float u, float v, const point3& p, _texture** textures, const void* object_data) {
    image* img = reinterpret_cast<image*>(const_cast<void*>(object_data));

    // Clamp u,v
    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    // Sample image texture
    if (img->channels == 1) {
        float value = tex2D<float>(img->tex, u, v);
        return color(value, value, value);
    } else if (img->channels == 4) {
        float4 value = tex2D<float4>(img->tex, u, v);
        return color(value.x, value.y, value.z);
    }
    return color(1, 0, 1); // should not get here
}

/**
 * @brief Get the color of a texture at a specified point
 * 
 * @param[in] u Texture coordinate u
 * @param[in] v Texture coordinate v
 * @param[in] p Hit point
 * @param[in] textures Textures in scene
 * @return Color of texture
 */
__device__ color _texture::value(float u, float v, const point3& p, _texture** textures) const {
    switch (type) {
        case texture_type::SOLID_COLOR:
            return value_solid_color(u, v, p, textures, this);
        case texture_type::CHECKER:
            return value_checker(u, v, p, textures, this);
        case texture_type::IMAGE:
            return value_image(u, v, p, textures, this);
        default:
            return color(0, 0, 0);
    }
}

/**
 * @brief Copy textures to device memory
 * 
 * @param[in] textures Host textures
 * @return Device textures
 */
_texture** allocate_textures(const std::vector<_texture*>& textures) {
    _texture** d_textures;
    size_t texture_count = textures.size();
    cudaMalloc(&d_textures, texture_count * sizeof(_texture*));
    
    // Allocate textures on device
    std::vector<_texture*> temp_vector(texture_count);
    for (int i = 0; i < textures.size(); i++) {
        _texture* d_tex = nullptr;
        if (auto sc = dynamic_cast<solid_color*>(textures[i])) {
            d_tex = copy_object(*sc);
        } else if (auto ch = dynamic_cast<checker*>(textures[i])) {
            d_tex = copy_object(*ch);
        } else if (auto img = dynamic_cast<image*>(textures[i])) {
            d_tex = copy_object(*img);
        }
        temp_vector[i] = d_tex;
    }
    checkCudaErrors(cudaMemcpy(d_textures, temp_vector.data(), texture_count * sizeof(_texture*), cudaMemcpyHostToDevice));

    return d_textures;
}

/**
 * @brief Free textures from device memory
 * 
 * @param[in] d_textures Device textures
 * @param[in] texture_count Number of textures to free
 */
void free_textures(_texture** d_textures, int texture_count) {
    // Copy pointer array
    _texture** h_textures = new _texture*[texture_count];
    checkCudaErrors(cudaMemcpy(h_textures, d_textures, texture_count * sizeof(_texture*), cudaMemcpyDeviceToHost));

    // Copy each texture to host
    for (int i = 0; i < texture_count; i++) {

        texture_type tex_type;
        checkCudaErrors(cudaMemcpy(&tex_type, &h_textures[i]->type, sizeof(texture_type), cudaMemcpyDeviceToHost));
        
        if (tex_type == texture_type::IMAGE) {
            image* h_image = new image();
            checkCudaErrors(cudaMemcpy(h_image, h_textures[i], sizeof(image), cudaMemcpyDeviceToHost));
            delete h_image;
        } else if (tex_type == texture_type::CHECKER) {
            checker* h_checker = new checker();
            checkCudaErrors(cudaMemcpy(h_checker, h_textures[i], sizeof(checker), cudaMemcpyDeviceToHost));
            delete h_checker;
        } else if (tex_type == texture_type::SOLID_COLOR) {
            solid_color* h_solid_color = new solid_color();
            checkCudaErrors(cudaMemcpy(h_solid_color, h_textures[i], sizeof(solid_color), cudaMemcpyDeviceToHost));
            delete h_solid_color;
        }
        checkCudaErrors(cudaFree(h_textures[i]));
    }

    // Free array
    checkCudaErrors(cudaFree(d_textures));
    delete[] h_textures;
}

/**
 * @brief Print texture
 * 
 * @param out Output stream
 * @param tex Texture to print
 * @return output stream
 */
inline std::ostream& operator<<(std::ostream& out, const _texture* tex) {
    if (tex->type == texture_type::SOLID_COLOR) {
        solid_color* sc = (solid_color*)tex;
        return out << "<solid_color albedo=(" << sc->albedo << ")>";
    } else if (tex->type == texture_type::CHECKER) {
        checker* ch = (checker*)tex;
        return out << "<checker odd=" << ch->odd_idx << " even=" << ch->even_idx << " inv_scale=" << ch->inv_scale << ">";
    } else if (tex->type == texture_type::IMAGE) {
        image* img = (image*)tex;
        return out << "<image width=" << img->width << " height=" << img->height << " channels=" << img->channels << ">";
    } else {
        return out << "<Unknown texture>";
    }
}

#endif
