/**
 * @file vec2.cuh
 * @author Jasper Jeuken
 * @brief Defines a 2D vector
 */
#ifndef VEC2_H
#define VEC2_H

#include <iostream>
#include "utility.cuh"

/**
 * @class vec2
 * @brief 2D vector
 */
class vec2 {
public:
    /**
     * @brief Default constructor
     * 
     * @return Zero vector
     */
    __host__ __device__ vec2() : e{0, 0} {}

    /**
     * @brief Construct a new 2D vector
     * 
     * @param[in] e0 X-value
     * @param[in] e1 Y-value
     * @return Constructed vector
     */
    __host__ __device__ vec2(float e0, float e1) : e{ e0, e1 } {}

    // Get functions
    __host__ __device__ inline float x() const { return e[0]; } ///< Get x-component
    __host__ __device__ inline float y() const { return e[1]; } ///< Get y-component
    __host__ __device__ inline float operator[](int i) const { return e[i]; } ///< Get i-th component (`0=x`, `1=y`)
    __host__ __device__ inline float& operator[](int i) { return e[i]; } ///< Get i-th component (`0=x`, `1=y`)

    __host__ __device__ inline const vec2& operator+() const { return *this; } ///< Unary plus operator
    __host__ __device__ inline const vec2 operator-() const { return vec2(-e[0], -e[1]); } ///< Negate vector values

    // Assignment overload
    __host__ __device__ inline vec2& vec2::operator+=(const vec2& other);
    __host__ __device__ inline vec2& vec2::operator-=(const vec2& other);
    __host__ __device__ inline vec2& vec2::operator*=(const vec2& other);
    __host__ __device__ inline vec2& vec2::operator/=(const vec2& other);
    __host__ __device__ inline vec2& vec2::operator*=(const float scalar);
    __host__ __device__ inline vec2& vec2::operator/=(const float scalar);

    // Operator overload
    __host__ __device__ friend inline vec2 operator+(const vec2& u, const vec2& v);
    __host__ __device__ friend inline vec2 operator-(const vec2& u, const vec2& v);
    __host__ __device__ friend inline vec2 operator*(const vec2& u, const vec2& v);
    __host__ __device__ friend inline vec2 operator/(const vec2& u, const vec2& v);
    __host__ __device__ friend inline vec2 operator+(const vec2& u, const float scalar);
    __host__ __device__ friend inline vec2 operator-(const vec2& u, const float scalar);
    __host__ __device__ friend inline vec2 operator*(const vec2& u, const float scalar);
    __host__ __device__ friend inline vec2 operator/(const vec2& u, const float scalar);
    __host__ __device__ friend inline vec2 operator*(const float scalar, const vec2& u);

    /**
     * @brief Get the square of the length of the vector
     * 
     * @return Square of the vector length
     */
    __host__ __device__ inline float length_squared() const {
        return e[0] * e[0] + e[1] * e[1];
    }

    /**
     * @brief Get the length of the vector
     * 
     * @return Vector length
     */
    __host__ __device__ inline float length() const {
        return std::sqrtf(length_squared());
    }

    /**
     * @brief Check if the vector is near-zero
     * 
     * @param[in] epsilon Bounds for near-zero (default `1e-8`)
     * @return Whether vector is near-zero
     */
    __host__ __device__ bool near_zero(float epsilon = 1e-8f) const {
        return (fabsf(e[0]) < epsilon) && (fabsf(e[1]) < epsilon);
    }

    /**
     * @brief Element-wise square root
     * 
     * @param[in] vec Vector to take element-wise square root of
     * @return Vector with square root of elements
     */
    __host__ __device__ inline static vec2 sqrt(const vec2& vec) {
        return vec2(std::sqrtf(vec.e[0]), std::sqrtf(vec.e[1]));
    }

    /**
     * @brief Calculate the dot product of two vectors
     * 
     * @param[in] u First vector
     * @param[in] v Second vector
     * @return Dot product
     */
    __host__ __device__ inline static float dot(const vec2& u, const vec2& v) {
        return u.e[0] * v.e[0] + u.e[1] * v.e[1];
    }

    /**
     * @brief Create a unit vector from a vector (length = 1)
     * 
     * @param[in] vec Vector
     * @return Unit vector
     */
    __host__ __device__ inline static vec2 unit_vector(const vec2& vec) {
        return vec / vec.length();
    }

public:
    float e[2]; ///< Vector data
};

/**
 * @brief Print a 2D vector
 * 
 * @param[in] out Output stream
 * @param[in] vec Vector to print
 * @return Output stream
 */
inline std::ostream& operator<<(std::ostream& out, const vec2& vec) {
    return out << vec.e[0] << " " << vec.e[1];
}

/**
 * @brief Read a 2D vector
 * 
 * @param[in] in Input stream
 * @param[out] vec Vector to read to
 * @return Input stream
 */
inline std::istream& operator>>(std::istream& in, vec2& vec) {
    in >> vec.e[0] >> vec.e[1];
    return in;
}

/**
 * @brief Add another vector
 * 
 * @param[in] other Other vector
 * @return Sum of vectors
 */
__host__ __device__ inline vec2& vec2::operator+=(const vec2& other) {
    e[0] += other.e[0];
    e[1] += other.e[1];
    return *this;
}

/**
 * @brief Subtract another vector
 * 
 * @param[in] other Other vector
 * @return Difference of vectors
 */
__host__ __device__ inline vec2& vec2::operator-=(const vec2& other) {
    e[0] -= other.e[0];
    e[1] -= other.e[1];
    return *this;
}

/**
 * @brief Multiply with another vector
 * 
 * @param[in] other Other vector
 * @return Product of vectors
 */
__host__ __device__ inline vec2& vec2::operator*=(const vec2& other) {
    e[0] *= other.e[0];
    e[1] *= other.e[1];
    return *this;
}

/**
 * @brief Divide by another vector
 * 
 * @param[in] other Other vector
 * @return Quotient of vectors
 */
__host__ __device__ inline vec2& vec2::operator/=(const vec2& other) {
    e[0] /= other.e[0];
    e[1] /= other.e[1];
    return *this;
}

/**
 * @brief Multiply with a scalar
 * 
 * @param[in] scalar Scalar value
 * @return Product of vector and scalar
 */
__host__ __device__ inline vec2& vec2::operator*=(const float scalar) {
    e[0] *= scalar;
    e[1] *= scalar;
    return *this;
}

/**
 * @brief Divide by ascalar
 * 
 * @param[in] scalar Scalar value
 * @return Quotient of vector and scalar
 */
__host__ __device__ inline vec2& vec2::operator/=(const float scalar) {
    e[0] /= scalar;
    e[1] /= scalar;
    return *this;
}

/**
 * @brief Add two vectors
 * 
 * @param[in] u First vector
 * @param[in] v Second vector
 * @return Sum of vectors
 */
__host__ __device__ inline vec2 operator+(const vec2& u, const vec2& v) {
    return vec2(u.e[0] + v.e[0], u.e[1] + v.e[1]);
}

/**
 * @brief Add vector and scalar
 * 
 * @param[in] u Vector
 * @param[in] scalar Scalar value
 * @return Sum of vector and scalar
 */
__host__ __device__ inline vec2 operator+(const vec2& u, const float scalar) {
    return vec2(u.e[0] + scalar, u.e[1] + scalar);
}

/**
 * @brief Subtract two vectors
 * 
 * @param[in] u First vector
 * @param[in] v Second vector
 * @return Difference of vectors
 */
__host__ __device__ inline vec2 operator-(const vec2& u, const vec2& v) {
    return vec2(u.e[0] - v.e[0], u.e[1] - v.e[1]);
}

/**
 * @brief Subtract vector and scalar
 * 
 * @param[in] u Vector
 * @param[in] scalar Scalar value
 * @return Difference of vector and scalar
 */
__host__ __device__ inline vec2 operator-(const vec2& u, const float scalar) {
    return vec2(u.e[0] - scalar, u.e[1] - scalar);
}

/**
 * @brief Multiply two vectors
 * 
 * @param[in] u First vector
 * @param[in] v Second vector
 * @return Product of vectors
 */
__host__ __device__ inline vec2 operator*(const vec2& u, const vec2& v) {
    return vec2(u.e[0] * v.e[0], u.e[1] * v.e[1]);
}

/**
 * @brief Multiply vector and scalar
 * 
 * @param[in] u Vector
 * @param[in] scalar Scalar value
 * @return Product of vector and scalar
 */
__host__ __device__ inline vec2 operator*(const vec2& u, const float scalar) {
    return vec2(u.e[0] * scalar, u.e[1] * scalar);
}

/**
 * @brief Multiply scalar and vector
 * 
 * @param[in] scalar Scalar value
 * @param[in] u Vector
 * @return Product of scalar and vector
 */
__host__ __device__ inline vec2 operator*(const float scalar, const vec2& u) {
    return vec2(u.e[0] * scalar, u.e[1] * scalar);
}

/**
 * @brief Divide two vectors
 * 
 * @param[in] u First vectors
 * @param[in] v Second vector
 * @return Quotient of vectors
 */
__host__ __device__ inline vec2 operator/(const vec2& u, const vec2& v) {
    return vec2(u.e[0] / v.e[0], u.e[1] / v.e[1]);
}

/**
 * @brief Divide vector by scalar
 * 
 * @param[in] u Vector
 * @param[in] scalar Scalar value
 * @return Quotient of vector and scalar 
 */
__host__ __device__ inline vec2 operator/(const vec2& u, const float scalar) {
    return vec2(u.e[0] / scalar, u.e[1] / scalar);
}

/**
 * @brief Parse a required 2D vector from a YAML node
 * 
 * @param[in] node YAML node
 * @param[in] field Field name
 * @param[in] error_message Error message if field is not found
 * @return Parsed 2D vector
 */
vec2 parse_required_vec2(const YAML::Node& node, const std::string& field, const std::string& error_message) {
    try {
        const YAML::Node& vec_node = node[field];
        return vec2(vec_node[0].as<float>(), vec_node[1].as<float>());
    } catch (YAML::Exception& e) {
        error_with_message(error_message);
    }
    return vec2(); // Unreachable
}

/**
 * @brief Parse an optional 2D vector from a YAML node
 * 
 * @param[in] node YAML node
 * @param[in] field Field name
 * @param[in] default_value Default value if field is not found
 * @return Parsed 2D vector
 */
vec2 parse_optional_vec2(const YAML::Node& node, const std::string& field, vec2 default_value) {
    try {
        const YAML::Node& vec_node = node[field];
        if (!vec_node) {
            return default_value;
        }
        return vec2(vec_node[0].as<float>(), vec_node[1].as<float>());
    } catch (YAML::Exception& e) {
        error_with_message("Field '" + field + "' is invalid");
    }
    return vec2(); // Unreachable
}

#endif