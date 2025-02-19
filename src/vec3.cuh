/**
 * @file vec3.cuh
 * @author Jasper Jeuken
 * @brief Defines a 3D vector
 */
#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <iostream>
#include <curand_kernel.h>

#include "utility.cuh"

/**
 * @class vec3
 * @brief 3D vector
 */
class vec3 {

public:
/**
     * @brief Default constructor
     * 
     * @return Zero vector
     */
    __host__ __device__ vec3() : e{0, 0, 0} {}

    /**
     * @brief Construct a new 3D vector
     * 
     * @param[in] e0 X-value
     * @param[in] e1 Y-value
     * @param[in] e2 Z-value
     * @return Constructed vector
     */
    __host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

    // Get functions
    __host__ __device__ inline float x() const { return e[0]; } ///< Get x-component
    __host__ __device__ inline float y() const { return e[1]; } ///< Get y-component
    __host__ __device__ inline float z() const { return e[2]; } ///< Get z-component
    __host__ __device__ inline float r() const { return e[0]; } ///< Get x-component
    __host__ __device__ inline float g() const { return e[1]; } ///< Get y-component
    __host__ __device__ inline float b() const { return e[2]; } ///< Get z-component
    __host__ __device__ inline float operator[](int i) const { return e[i]; } ///< Get i-th component (`0=x`, `1=y`, `2=z`)
    __host__ __device__ inline float& operator[](int i) { return e[i]; } ///< Get i-th component (`0=x`, `1=y`, `2=z`)
    __host__ __device__ inline float3 as_float3() const { return make_float3(e[0], e[1], e[2]); } ///< Get float3 representation

    __host__ __device__ inline const vec3& operator+() const { return *this; } ///< Unary plus operator
    __host__ __device__ inline const vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); } ///< Negate vector values

    // Assignment overload
    __host__ __device__ inline vec3& vec3::operator+=(const vec3& other);
    __host__ __device__ inline vec3& vec3::operator-=(const vec3& other);
    __host__ __device__ inline vec3& vec3::operator*=(const vec3& other);
    __host__ __device__ inline vec3& vec3::operator/=(const vec3& other);
    __host__ __device__ inline vec3& vec3::operator*=(const float scalar);
    __host__ __device__ inline vec3& vec3::operator/=(const float scalar);

    // Operator overload
    __host__ __device__ friend inline vec3 operator+(const vec3& u, const vec3& v);
    __host__ __device__ friend inline vec3 operator-(const vec3& u, const vec3& v);
    __host__ __device__ friend inline vec3 operator*(const vec3& u, const vec3& v);
    __host__ __device__ friend inline vec3 operator/(const vec3& u, const vec3& v);
    __host__ __device__ friend inline vec3 operator+(const vec3& u, const float scalar);
    __host__ __device__ friend inline vec3 operator-(const vec3& u, const float scalar);
    __host__ __device__ friend inline vec3 operator*(const vec3& u, const float scalar);
    __host__ __device__ friend inline vec3 operator/(const vec3& u, const float scalar);
    __host__ __device__ friend inline vec3 operator*(const float scalar, const vec3& u);

    /**
     * @brief Get the square of the length of the vector
     * 
     * @return Square of the vector length
     */
    __host__ __device__ inline float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    /**
     * @brief Get the length of the vector
     * 
     * @return Vector length
     */
    __host__ __device__ inline float length() const {
        return std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    }

    /**
     * @brief Check if the vector is near-zero
     * 
     * @param[in] epsilon Bounds for near-zero (default `1e-8`)
     * @return Whether vector is near-zero
     */
    __host__ __device__ bool near_zero() const {
        const float s = 1e-8;
        return (abs(e[0]) < s) && (abs(e[1]) < s) && (abs(e[2]) < s);
    }

    /**
     * @brief Element-wise square root
     * 
     * @param[in] vec Vector to take element-wise square root of
     * @return Vector with square root of elements
     */
    __host__ __device__ inline static vec3 sqrt(const vec3& vec) {
        return vec3(std::sqrt(vec.e[0]), std::sqrt(vec.e[1]), std::sqrt(vec.e[2]));
    }

    /**
     * @brief Calculate the dot product of two vectors
     * 
     * @param[in] u First vector
     * @param[in] v Second vector
     * @return Dot product
     */
    __host__ __device__ inline static float dot(const vec3& u, const vec3& v) {
        return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
    }

    /**
     * @brief Calculate the cross product of two vectors
     * 
     * @param[in] u First vector
     * @param[in] v Second vector
     * @return Cross product
     */
    __host__ __device__ inline static vec3 cross(const vec3& u, const vec3& v) {
        return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                    u.e[2] * v.e[0] - u.e[0] * v.e[2],
                    u.e[0] * v.e[1] - u.e[1] * v.e[0]);
    }

    /**
     * @brief Create a unit vector from a vector (length = 1)
     * 
     * @param[in] vec Vector
     * @return Unit vector
     */
    __host__ __device__ inline static vec3 unit_vector(const vec3& vec) {
        return vec / vec.length();
    }

    /**
     * @brief Reflect a vector using a normal direction
     * 
     * @param[in] v Vector to reflect
     * @param[in] n Normal direction
     * @return Reflected vector
     */
    __device__ inline static vec3 reflect(const vec3& v, const vec3& n) {
        return v - 2.0f * dot(v, n) * n;
    }

    /**
     * @brief Refract a vector using a normal direction and refractive indices
     * 
     * Refraction is calculated according to Snell's law.
     * 
     * @param[in] uv Vector to refract
     * @param[in] n Normal direction
     * @param[in] ni_over_nt Ratio of refractive indices
     * @return Refracted vector
     */
    __device__ inline static vec3 refract(const vec3& uv, const vec3& n, float ni_over_nt) {
        float cos_theta = minf(vec3::dot(-uv, n), 1.0f);
        vec3 r_out_perp = ni_over_nt * (uv + cos_theta * n);
        vec3 r_out_parallel = -std::sqrt(std::abs(1.0f - r_out_perp.length_squared())) * n;
        return r_out_perp + r_out_parallel;
    }

    /**
     * @brief Generate a random 3D vector
     * 
     * @param[in] rand_state Random state
     * @return Random 3D vector
     */
    __device__ inline static vec3 random(curandState* rand_state) {
        return vec3(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state));
    }

    /**
     * @brief Generate a random 3D vector between a minimum and maximum value
     * 
     * @param[in] rand_state Random state
     * @param[in] min Minimum value
     * @param[in] max Maximum value
     * @return Random 3D vector in given range
     */
    __device__ inline static vec3 random(curandState* rand_state, float min, float max) {
        float x = min + (max - min) * curand_uniform(rand_state);
        float y = min + (max - min) * curand_uniform(rand_state);
        float z = min + (max - min) * curand_uniform(rand_state);
        return vec3(x, y, z);
    }

    /**
     * @brief Generate a random unit vector
     * 
     * @param[in] rand_state Random state
     * @return Random 3D unit vector
     */
    __device__ inline static vec3 random_unit_vector(curandState* rand_state) {
        while (true) {
            vec3 p = vec3::random(rand_state, -1.0f, 1.0f);
            float lensq = p.length_squared();
            if (1e-30f < lensq && lensq <= 1) {
                return vec3::unit_vector(p);
            }
        }
    }

    /**
     * @brief Generate a random vector on a hemisphere
     * 
     * @param[in] rand_state Random state
     * @param[in] normal Hemisphere normal direction
     * @return Random 3D vector on hemisphere
     */
    __device__ inline static vec3 random_on_hemisphere(curandState* rand_state, const vec3& normal) {
        vec3 on_unit_sphere = random_unit_vector(rand_state);
        if (vec3::dot(on_unit_sphere, normal) > 0.0f) {
            return on_unit_sphere;
        }
        return -on_unit_sphere;
    }

    /**
     * @brief Generate a random vector on a sphere
     * 
     * @param[in] rand_state Random state
     * @return Random 3D vector on sphere
     */
    __device__ inline static vec3 random_in_unit_sphere(curandState* rand_state) {
        vec3 p = vec3::random(rand_state, -1.0f, 1.0f);
        while (p.length_squared() >= 1.0f) {
            p = vec3::random(rand_state, -1.0f, 1.0f);
        }
        return p;
    }

    /**
     * @brief Generate a random vector on a unit disk
     * 
     * @param[in] rand_state Random state
     * @return Random 3D vector on unit disk 
     */
    __device__ inline static vec3 random_in_unit_disk(curandState* rand_state) {
        vec3 p = 2.0f * vec3(curand_uniform(rand_state), curand_uniform(rand_state), 0) - vec3(1, 1, 0);
        while(vec3::dot(p, p) >= 1.0f) {
            p = 2.0f * vec3(curand_uniform(rand_state), curand_uniform(rand_state), 0) - vec3(1, 1, 0);
        }
        return p;
    }

    /**
     * @brief Generate a random vector (host only)
     * 
     * @return Random 3D vector 
     */
    __host__ inline static vec3 host_random() {
        return vec3(random_float(), random_float(), random_float());
    }

    /**
     * @brief Generate a random vector between a minimum and maximum value (host only)
     * 
     * @param[in] min Minimum value
     * @param[in] max Maximum value
     * @return Random 3D vector in given range
     */
    __host__ inline static vec3 host_random(float min, float max) {
        return vec3(random_float(min, max), random_float(min, max), random_float(min, max));
    }
   
public:
    float e[3]; ///< Vector data
};

// Aliases
using point3 = vec3;
using color = vec3;

/**
 * @brief Print a 3D vector
 * 
 * @param[in] out Output stream
 * @param[in] vec Vector to print
 * @return Output stream
 */
inline std::ostream& operator<<(std::ostream& out, const vec3& vec) {
    return out << vec.e[0] << " " << vec.e[1] << " " << vec.e[2];
}

/**
 * @brief Read a 3D vector
 * 
 * @param[in] in Input stream
 * @param[out] vec Vector to read to
 * @return Input stream
 */
inline std::istream& operator>>(std::istream& in, vec3& vec) {
    in >> vec.e[0] >> vec.e[1] >> vec.e[2];
    return in;
}

/**
 * @brief Add another vector
 * 
 * @param[in] other Other vector
 * @return Sum of vectors
 */
__host__ __device__ inline vec3& vec3::operator+=(const vec3& other) {
    e[0] += other.e[0];
    e[1] += other.e[1];
    e[2] += other.e[2];
    return *this;
}

/**
 * @brief Subtract another vector
 * 
 * @param[in] other Other vector
 * @return Difference of vectors
 */
__host__ __device__ inline vec3& vec3::operator-=(const vec3& other) {
    e[0] -= other.e[0];
    e[1] -= other.e[1];
    e[2] -= other.e[2];
    return *this;
}

/**
 * @brief Multiply with another vector
 * 
 * @param[in] other Other vector
 * @return Product of vectors
 */
__host__ __device__ inline vec3& vec3::operator*=(const vec3& other) {
    e[0] *= other.e[0];
    e[1] *= other.e[1];
    e[2] *= other.e[2];
    return *this;
}

/**
 * @brief Divide by another vector
 * 
 * @param[in] other Other vector
 * @return Quotient of vectors
 */
__host__ __device__ inline vec3& vec3::operator/=(const vec3& other) {
    e[0] /= other.e[0];
    e[1] /= other.e[1];
    e[2] /= other.e[2];
    return *this;
}

/**
 * @brief Multiply with a scalar
 * 
 * @param[in] scalar Scalar value
 * @return Product of vector and scalar
 */
__host__ __device__ inline vec3& vec3::operator*=(const float scalar) {
    e[0] *= scalar;
    e[1] *= scalar;
    e[2] *= scalar;
    return *this;
}

/**
 * @brief Divide by ascalar
 * 
 * @param[in] scalar Scalar value
 * @return Quotient of vector and scalar
 */
__host__ __device__ inline vec3& vec3::operator/=(const float scalar) {
    e[0] /= scalar;
    e[1] /= scalar;
    e[2] /= scalar;
    return *this;
}

/**
 * @brief Add two vectors
 * 
 * @param[in] u First vector
 * @param[in] v Second vector
 * @return Sum of vectors
 */
__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

/**
 * @brief Add vector and scalar
 * 
 * @param[in] u Vector
 * @param[in] scalar Scalar value
 * @return Sum of vector and scalar
 */
__host__ __device__ inline vec3 operator+(const vec3& u, const float scalar) {
    return vec3(u.e[0] + scalar, u.e[1] + scalar, u.e[2] + scalar);
}

/**
 * @brief Subtract two vectors
 * 
 * @param[in] u First vector
 * @param[in] v Second vector
 * @return Difference of vectors
 */
__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

/**
 * @brief Subtract vector and scalar
 * 
 * @param[in] u Vector
 * @param[in] scalar Scalar value
 * @return Difference of vector and scalar
 */
__host__ __device__ inline vec3 operator-(const vec3& u, const float scalar) {
    return vec3(u.e[0] - scalar, u.e[1] - scalar, u.e[2] - scalar);
}

/**
 * @brief Multiply two vectors
 * 
 * @param[in] u First vector
 * @param[in] v Second vector
 * @return Product of vectors
 */
__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

/**
 * @brief Multiply vector and scalar
 * 
 * @param[in] u Vector
 * @param[in] scalar Scalar value
 * @return Product of vector and scalar
 */
__host__ __device__ inline vec3 operator*(const vec3& u, const float scalar) {
    return vec3(u.e[0] * scalar, u.e[1] * scalar, u.e[2] * scalar);
}

/**
 * @brief Multiply scalar and vector
 * 
 * @param[in] scalar Scalar value
 * @param[in] u Vector
 * @return Product of scalar and vector
 */
__host__ __device__ inline vec3 operator*(const float scalar, const vec3& u) {
    return vec3(u.e[0] * scalar, u.e[1] * scalar, u.e[2] * scalar);
}

/**
 * @brief Divide two vectors
 * 
 * @param[in] u First vectors
 * @param[in] v Second vector
 * @return Quotient of vectors
 */
__host__ __device__ inline vec3 operator/(const vec3& u, const vec3& v) {
    return vec3(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}

/**
 * @brief Divide vector by scalar
 * 
 * @param[in] u Vector
 * @param[in] scalar Scalar value
 * @return Quotient of vector and scalar 
 */
__host__ __device__ inline vec3 operator/(const vec3& u, const float scalar) {
    return vec3(u.e[0] / scalar, u.e[1] / scalar, u.e[2] / scalar);
}

/**
 * @brief Convert a float [0,1] to an int [0,255]
 * 
 * @param value Value to convert
 * @return Converted value
 */
int color_value(float value) {
    return static_cast<int>(256 * clamp(value, 0.0f, 0.9999999f));
}

/**
 * @brief Parse a required 3D vector from a YAML node
 * 
 * @param[in] node YAML node
 * @param[in] field Field name
 * @param[in] error_message Error message if field is not found
 * @return Parsed 3D vector
 */
vec3 parse_required_vec3(const YAML::Node& node, const std::string& field, const std::string& error_message) {
    try {
        const YAML::Node& vec_node = node[field];
        return vec3(vec_node[0].as<float>(), vec_node[1].as<float>(), vec_node[2].as<float>());
    } catch (YAML::Exception& e) {
        error_with_message(error_message);
    }
    return vec3(); // Unreachable
}

/**
 * @brief Parse an optional 3D vector from a YAML node
 * 
 * @param[in] node YAML node
 * @param[in] field Field name
 * @param[in] default_value Default value if field is not found
 * @return Parsed 3D vector
 */
vec3 parse_optional_vec3(const YAML::Node& node, const std::string& field, vec3 default_value) {
    try {
        const YAML::Node& vec_node = node[field];
        if (!vec_node) {
            return default_value;
        }
        return vec3(vec_node[0].as<float>(), vec_node[1].as<float>(), vec_node[2].as<float>());
    } catch (YAML::Exception& e) {
        error_with_message("Field '" + field + "' is invalid");
    }
    return vec3(); // Unreachable
}

#endif