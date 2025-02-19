/**
 * @file utility.cuh
 * @author Jasper Jeuken
 * @brief Defines utility functions and constants
 */
#ifndef UTILITY_H
#define UTILITY_H

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <iostream>
#include <yaml-cpp/yaml.h>


using std::shared_ptr;
using std::make_shared;


// Constants
#define CONST_PI 3.1415926535897932385f
const float h_pi = CONST_PI;
__constant__ const float d_pi = CONST_PI;
#undef CONST_PI
#define CONST_INFINITY std::numeric_limits<float>::infinity()
const float h_infinity = CONST_INFINITY;
__constant__ const float d_infinity = CONST_INFINITY;
#undef CONST_INFINITY

/**
 * @brief Clamp a value between a minimum and maximum
 * 
 * @param[in] x Value to clamp
 * @param[in] min Minimum value
 * @param[in] max Maximum value
 * @return Clamped value
 */
__host__ __device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

/**
 * @brief Map a value from one range to another
 * 
 * @param[in] value Value to map
 * @param[in] min0 Start of original range
 * @param[in] max0 End of original range
 * @param[in] min1 Start of new range
 * @param[in] max1 End of new range
 * @return Mapped value
 */
__host__ __device__ inline float map(float value, float min0, float max0, float min1, float max1) {
    return min1 + (value - min0) * (max1 - min1) / (max0 - min0);
}

/**
 * @brief Select smallest value
 * 
 * @param[in] a First value
 * @param[in] b Second value
 * @return Smallest value
 */
__host__ __device__ inline float minf(float a, float b) {
    if (a < b) {
        return a;
    }
    return b;
}

/**
 * @brief Select biggest value
 * 
 * @param[in] a First value
 * @param[in] b Second value
 * @return Biggest value
 */
__host__ __device__ inline float maxf(float a, float b) {
    if (a > b) {
        return a;
    }
    return b;
}

/**
 * @brief Convert degrees to radians
 * 
 * @param[in] degrees Value in degrees
 * @return Value in radians
 */
__host__ __device__ inline float degrees_to_radians(float degrees) {
    #ifdef __CUDA_ARCH__
        return degrees * d_pi / 180.0f;
    #else
        return degrees * h_pi / 180.0f;
    #endif
}

/**
 * @brief Convert radians to degrees
 * 
 * @param[in] degrees Value in radians
 * @return Value in degrees
 */
__host__ __device__ inline float radians_to_degrees(float radians) {
    #ifdef __CUDA_ARCH__
        return radians * 180.0f / d_pi;
    #else
        return radians * 180.0f / h_pi;
    #endif
}

/**
 * @brief Generate a random float
 * 
 * @return Random float
 */
__host__ float random_float() {
    return std::rand() / (RAND_MAX + 1.0f);
}

/**
 * @brief Generate a random float between two values
 * 
 * @param[in] min Minimum value
 * @param[in] max Maximum value
 * @return Random value in given range
 */
__host__ float random_float(float min, float max) {
    return min + (max - min) * random_float();
}

/**
 * @brief Create a string with the current date/time (YYYYMMDD_HHMMSS)
 * 
 * @return Current date/time string
 */
__host__ std::string date_time_string() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

// Shorthand for check_cuda
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

/**
 * @brief Run a CUDA function with error handling
 * 
 * @param[in] result Function result (success/error)
 * @param[in] func Function to execute
 * @param[in] file File in which function is executed
 * @param[in] line Line at which function is executed
 */
__host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "Cuda error: " << static_cast<unsigned int>(result) << " at " <<
                     file << ": " << line << " '" << func << "' (" << cudaGetErrorString(result) << ")" << std::endl;
        
        cudaDeviceReset();
        exit(99);
    }
}

/**
 * @brief Copy an object to device memory
 * 
 * @tparam T Object type
 * @param[in] host_obj Object to copy
 * @return T* Device pointer
 */
template<typename T>
T* copy_object(const T& host_obj) {
    T* device_obj;
    checkCudaErrors(cudaMalloc(&device_obj, sizeof(T)));
    checkCudaErrors(cudaMemcpy(device_obj, &host_obj, sizeof(T), cudaMemcpyHostToDevice));
    return device_obj;
}

/**
 * @brief Copy an array to device memory
 * 
 * @tparam T Array element type
 * @param[in] host_obj Array to copy
 * @param[in] size Array size
 * @return Device pointer
 */
template<typename T>
T* copy_object(const T& host_obj, size_t size) {
    T* device_obj;
    checkCudaErrors(cudaMalloc(&device_obj, sizeof(T) * size));
    checkCudaErrors(cudaMemcpy(device_obj, &host_obj, sizeof(T) * size, cudaMemcpyHostToDevice));
    return device_obj;
}

/**
 * @brief Copy an object to host memory
 * 
 * @tparam T Object type
 * @param[in] dev_obj Device pointer
 * @return Host pointer
 */
template<typename T>
T* copy_object_to_host(const T& dev_obj) {
    T* host_obj = new T;
    checkCudaErrors(cudaMemcpy(host_obj, &dev_obj, sizeof(T), cudaMemcpyDeviceToHost));
    return host_obj;
}

/**
 * @brief Free an object in device memory
 * 
 * @tparam T Object type
 * @param[in] device_ptr Device pointer
 */
template<typename T>
void free_object(T* device_ptr) {
    checkCudaErrors(cudaFree(device_ptr));
}

/**
 * @brief Free an array in device memory
 * 
 * @tparam T Array element type
 * @param[in] device_array Device pointer
 * @param[in] array_size Array size
 */
template<typename T>
void free_pointer_array(T** device_array, size_t array_size) {
    T* temp_ptr;
    for (size_t i = 0; i < array_size; i++) {
        checkCudaErrors(cudaMemcpy(&temp_ptr, device_array + i, sizeof(T*), cudaMemcpyDeviceToHost));
        if (temp_ptr) {
            free_object(temp_ptr);
        }
    }
    free_object(device_array);
}

/**
 * @brief Print eror message, then exit with error
 * 
 * @param[in] message Error message
 */
void error_with_message(std::string message) {
    std::cerr << message << std::endl;
    exit(1);
}

/**
 * @brief Check if a value is one (or more) of the specified options
 * 
 * @tparam T Object type
 * @tparam Opts Option type
 * @param[in] val Value to check against options
 * @param[in] opts Predefined options
 * @return Whether value is in options
 */
template<typename T, typename ...Opts>
bool any_of(T val, Opts ...opts) {
    return (... || (val == opts));
}

/**
 * @brief Parse a required value from a YAML node
 * 
 * @tparam T Value type
 * @param[in] node YAML node
 * @param[in] field Field name
 * @param[in] error_message Error message if field is not found
 * @return Parsed value
 */
template <typename T>
T parse_required(const YAML::Node& node, const std::string& field, const std::string& error_message) {
    try {
        return node[field].as<T>();
    } catch (YAML::Exception& e) {
        error_with_message(error_message);
    }
    return T(); // Unreachable
}

/**
 * @brief Parse optional value from a YAML node
 * 
 * @tparam T Value type
 * @param[in] node YAML node
 * @param[in] field Field name
 * @param[in] default_value Default value if field is not found
 * @return Parsed value
 */
template <typename T>
T parse_optional(const YAML::Node& node, const std::string& field, T default_value) {
    const YAML::Node& field_node = node[field];
    if (!field_node) {
        return default_value;
    }
    try {
        return field_node.as<T>();
    } catch (YAML::Exception& e) {
        error_with_message("Field '" + field + "' is invalid");
    }
    return default_value; // Unreachable
}

#endif