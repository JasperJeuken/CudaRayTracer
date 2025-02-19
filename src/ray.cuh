/**
 * @file ray.cuh
 * @author Jasper Jeuken
 * @brief Defines a ray class
 */
#ifndef RAY_H
#define RAY_H

#include "vec3.cuh"

/**
 * @class ray
 * @brief Ray
 * 
 * A ray is defined by an origin and a direction vector.
 * The ray also has a time value.
 */
class ray {
public:
    /**
     * @brief Default constructor
     * 
     * @return Empty ray 
     */
    __host__ __device__ ray() {}

    /**
     * @brief Construct a new ray object from an origin, direction, and time
     * 
     * @param[in] _orig Ray origin
     * @param[in] _dir Ray direction
     * @param[in] _time Ray time
     * @return Constructed ray 
     */
    __host__ __device__ ray(const point3& _orig, const vec3& _dir, float _time) : orig(_orig), dir(_dir), tm(_time) {}
    
    /**
     * @brief Construct a new ray object from an origin and direction
     * 
     * @param[in] _orig Ray origin
     * @param[in] _dir Ray direction
     * @return Constructed ray 
     */
    __host__ __device__ ray(const point3& _orig, const vec3& _dir) : ray(_orig, _dir, 0.0f) {}

    /// @brief Get the ray origin
    __host__ __device__ inline point3 origin() const { return orig; }

    /// @brief Get the ray direction
    __host__ __device__ inline vec3 direction() const { return dir; }

    /// @brief Get a point on the ray at a given time
    /// @param[in] t Time value
    /// @return Point on the ray at time t
    __host__ __device__ inline point3 at(float t) const { return orig + dir * t; }

    /// @brief Get the ray time
    __host__ __device__ inline float time() const { return tm; }

private:
    point3 orig; ///< Ray origin
    vec3 dir; ///< Ray direction
    float tm; ///< Ray time
};

#endif