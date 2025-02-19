/**
 * @file camera.cuh
 * @author Jasper Jeuken
 * @brief Defines the base camera class and derived subclasses
 */
#ifndef CAMERA_H
#define CAMERA_H

#include "ray.cuh"

/// @brief Enum for camera types
enum class camera_type {
    PERSPECTIVE,
    ORTHOGRAPHIC
};

/**
 * @struct render_information
 * @brief Information about the render
 */
struct render_information {
    int image_width; ///< Image width
    int image_height; ///< Image height
    int samples_per_pixel; ///< Samples per pixel
    int max_bounces; ///< Maximum number of bounces
    float gamma = 2.0f; ///< Gamma correction factor
};

/**
 * @struct camera_information
 * @brief Information about the camera
 */
struct camera_information {
    vec3 from; ///< Camera position
    vec3 to; ///< Camera target
    vec3 up; ///< Camera up vector
    float width; ///< Camera width (for orthographic only)
    float vfov; ///< Camera vertical field of view (for perspective only)
    float defocus_angle; ///< Camera defocus angle (for perspective only)
    float focus_dist; ///< Camera focus distance (for perspective only)
};

/**
 * @class camera
 * @brief Base camera
 */
class camera {
public:

    /**
     * @brief Construct a new perspective camera object
     * 
     * @param[in] _cam_info Camera information
     * @param[in] _render_info Render information
     * @return Constructed perspective camera object
     */
    __host__ camera(camera_information _cam_info, render_information _render_info) : type(camera_type::PERSPECTIVE), cam_info(_cam_info), render_info(_render_info) {
        aspect_ratio = float(render_info.image_width) / float(render_info.image_height);
        pixel_samples_scale = 1.0 / render_info.samples_per_pixel;
    }

    /**
     * @brief Construct a new camera object
     * 
     * @param[in] _cam_info Camera information
     * @param[in] _render_info Render information
     * @param[in] _type Camera type
     * @return Constructed camera object
     */
    __host__ camera(camera_information _cam_info, render_information _render_info, camera_type _type) : type(_type), cam_info(_cam_info), render_info(_render_info) {
        aspect_ratio = float(render_info.image_width) / float(render_info.image_height);
        pixel_samples_scale = 1.0 / render_info.samples_per_pixel;
    }

    __host__ virtual ~camera() {} ///< Virtual destructor

    /**
     * @brief Shoot a ray through a pixel
     * 
     * @param[in] i Pixel x-coordinate
     * @param[in] j Pixel y-coordinate
     * @param[in] rand_state Random state
     * @return Ray through the pixel
     */
    __device__ ray get_ray(int i, int j, curandState* rand_state) const;

    /**
     * @brief Sample a square
     * 
     * @param[in] rand_state Random state
     * @return Sampled square
     */
    __device__ vec3 sample_square(curandState* rand_state) const {
        vec3 v = vec3::random(rand_state, -0.5f, 0.5f);
        return v;
    }
    
    render_information render_info; ///< Render information
    camera_information cam_info; ///< Camera information
    camera_type type; ///< Camera type
    float aspect_ratio; ///< Image aspect ratio
    float pixel_samples_scale; ///< Pixel samples scale
};

/**
 * @class perspective_camera
 * @brief Perspective camera
 * 
 * A perspective camera is a camera that simulates the human eye. It has a field of view and a focus distance.
 */
class perspective_camera : public camera {
public:

    /**
     * @brief Construct a new perspective camera object
     * 
     * @param[in] _cam_info Camera information
     * @param[in] _render_info Render information
     * @return Constructed perspective camera object
     */
    __host__ perspective_camera(camera_information& _cam_info, render_information& _render_info) : camera(_cam_info, _render_info, camera_type::PERSPECTIVE) {
        center = cam_info.from;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(cam_info.vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * cam_info.focus_dist;
        auto viewport_width = viewport_height * (float(render_info.image_width)/render_info.image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = vec3::unit_vector(cam_info.from - cam_info.to);
        u = vec3::unit_vector(vec3::cross(cam_info.up, w));
        v = vec3::cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / render_info.image_width;
        pixel_delta_v = viewport_v / render_info.image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (cam_info.focus_dist * w) - viewport_u/2 - viewport_v/2;
        lower_left_corner = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = cam_info.focus_dist * std::tan(degrees_to_radians(cam_info.defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    /**
     * @brief Sample a disk
     * 
     * @param[in] rand_state Random state
     * @param[in] radius Disk radius
     * @return Sampled disk
     */
    __device__ vec3 sample_disk(curandState* rand_state, float radius) const {
        return radius * vec3::random_in_unit_disk(rand_state);
    }

    /**
     * @brief Sample a defocus disk
     * 
     * @param[in] rand_state Random state
     * @return Sampled defocus disk
     */
    __device__ point3 defocus_disk_sample(curandState* rand_state) const {
        vec3 v = vec3::random_in_unit_disk(rand_state);
        return center + (v[0] * defocus_disk_u) + (v[1] * defocus_disk_v);
    }

public:
    point3 center; ///< Camera center
    point3 lower_left_corner; ///< Viewport lower left corner
    vec3 pixel_delta_u; ///< Pixel step in u direction
    vec3 pixel_delta_v; ///< Pixel step in v direction
    vec3 u; ///< Camera u vector
    vec3 v; ///< Camera v vector
    vec3 w; ///< Camera w vector
    vec3 defocus_disk_u; ///< Defocus disk u vector
    vec3 defocus_disk_v; ///< Defocus disk v vector
};


/**
 * @class orthographic_camera
 * @brief Orthographic camera
 * 
 * An orthographic camera is a camera where the projection of the scene is parallel to the camera plane.
 */
class orthographic_camera : public camera {
public:

    /**
     * @brief Construct a new orthographic camera object
     * 
     * @param[in] _cam_info Camera information
     * @param[in] _render_info Render information
     * @return Constructed orthographic camera object
     */
    __host__ orthographic_camera(camera_information& _cam_info, render_information& _render_info) : camera(_cam_info, _render_info, camera_type::ORTHOGRAPHIC) {
        float viewport_width = cam_info.width;
        float viewport_height = viewport_width / aspect_ratio;

        w = - vec3::unit_vector(cam_info.from - cam_info.to);
        u = - vec3::unit_vector(vec3::cross(cam_info.up, w));
        v = - vec3::cross(w, u);

        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;

        pixel_delta_u = viewport_u / render_info.image_width;
        pixel_delta_v = viewport_v / render_info.image_height;

        vec3 viewport_upper_left = cam_info.from - viewport_u / 2 - viewport_v / 2;
        lower_left_corner = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

public:
    point3 lower_left_corner; ///< Viewport lower left corner
    vec3 u; ///< Camera u vector
    vec3 v; ///< Camera v vector
    vec3 w; ///< Camera w vector
    vec3 pixel_delta_u; ///< Pixel step in u direction
    vec3 pixel_delta_v; ///< Pixel step in v direction
};

/**
 * @brief Shoot a perspective ray through a pixel
 * 
 * @param[in] i Pixel x-coordinate
 * @param[in] j Pixel y-coordinate
 * @param[in] rand_state Random state
 * @param[in] object_data Camera object data
 * @return __device__ 
 */
__device__ ray perspective_ray(int i, int j, curandState* rand_state, const void* object_data) {
    perspective_camera* cam = (perspective_camera*)object_data;
    vec3 offset = cam->sample_square(rand_state);
    point3 pixel_sample = cam->lower_left_corner + (i + offset.x()) * cam->pixel_delta_u + (j + offset.y()) * cam->pixel_delta_v;
    point3 ray_origin = (cam->cam_info.defocus_angle <= 0) ? cam->center : cam->defocus_disk_sample(rand_state);
    vec3 ray_direction = pixel_sample - ray_origin;
    float ray_time = curand_uniform(rand_state);
    return ray(ray_origin, ray_direction, ray_time);
}

/**
 * @brief Shoot an orthographic ray through a pixel
 * 
 * @param[in] i Pixel x-coordinate
 * @param[in] j Pixel y-coordinate
 * @param[in] rand_state Random state
 * @param[in] object_data Camera object data
 * @return __device__ 
 */
__device__ ray orthographic_ray(int i, int j, curandState* rand_state, const void* object_data) {
    orthographic_camera* cam = (orthographic_camera*)object_data;
    vec3 offset = cam->sample_square(rand_state);
    point3 ray_origin = cam->lower_left_corner + (i + offset.x()) * cam->pixel_delta_u + (j + offset.y()) * cam->pixel_delta_v;
    vec3 ray_direction = cam->w;
    float ray_time = curand_uniform(rand_state);
    return ray(ray_origin, ray_direction, ray_time);
}

// Implementation of camera member function
__device__ ray camera::get_ray(int i, int j, curandState* rand_state) const {
    switch (type) {
        case camera_type::PERSPECTIVE:
            return perspective_ray(i, j, rand_state, this);
        case camera_type::ORTHOGRAPHIC:
            return orthographic_ray(i, j, rand_state, this);
        default:
            return ray(point3(0, 0, 0), vec3(0, 0, 0));
    }
}

/**
 * @brief Copy a camera object to device memory
 * 
 * @param[in] cam Camera object to copy
 * @return Device pointer to copied camera object
 */
camera* allocate_camera(camera* cam) {
    if (auto p = dynamic_cast<perspective_camera*>(cam)) {
        return copy_object(*p);
    } else if (auto o = dynamic_cast<orthographic_camera*>(cam)) {
        return copy_object(*o);
    } else {
        throw std::runtime_error("Unknown camera type");
    }
}

#endif