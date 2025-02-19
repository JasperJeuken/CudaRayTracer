/**
 * @file interval.cuh
 * @author Jasper Jeuken
 * @brief Defines an interval class
 */
#ifndef INTERVAL_H
#define INTERVAL_H

#include "utility.cuh"


/**
 * @class interval
 * @brief Closed interval [min, max]
 */
class interval {
public:
    float min; ///< The minimum value of the interval
    float max; ///< The maximum value of the interval

    /**
     * @brief Default constructor
     * @return Empty interval
     */
    __host__ __device__ interval() {
        #ifdef __CUDA_ARCH__
            min = d_infinity;
            max = -d_infinity;
        #else
            min = h_infinity;
            max = -h_infinity;
        #endif
    }

    /**
     * @brief Construct a new interval object from a minimum and maximum value
     * 
     * @param[in] _min The minimum value of the interval
     * @param[in] _max The maximum value of the interval
     * @return Interval between min and max
     */
    __host__ __device__ interval(float _min, float _max) : min(_min), max(_max) {}

    /**
     * @brief Construct a new interval object from two intervals
     * 
     * @param[in] a First interval
     * @param[in] b Second interval
     * @return Interval that contains both a and b
     */
    __host__ __device__ interval(const interval& a, const interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    /// @brief Get the size of the interval
    /// @return Interval size
    __host__ __device__ float size() const { return max - min; }

    /// @brief Check if the interval contains a value (inclusive)
    /// @param[in] x Value to check
    /// @return Whether the interval contains x
    __host__ __device__ bool contains(float x) const { return min <= x && x <= max; }

    /// @brief Check if the interval surrounds a value (exclusive)
    /// @param[in] x Value to check
    /// @return Whether the interval surrounds x
    __host__ __device__ bool surrounds(float x) const { return min < x && x < max; }

    /// @brief Clamp a value to the interval
    /// @param[in] x Value to clamp
    /// @return Clamped value
    __host__ __device__ float clamp(float x) const { return ::clamp(x, min, max); }

    /// @brief Expand the interval by a value
    /// @param[in] x Value to expand by
    /// @return Expanded interval
    __host__ __device__ interval expand(float x) const { return interval(min - x, max + x); }

    /// @brief Get the center of the interval
    /// @return Center of the interval
    __host__ __device__ float center() const { return 0.5f * (min + max); }

    static const interval empty; ///< Empty interval
    static const interval universe; ///< Universe interval
};

// Implementation of static members
const interval interval::empty = interval(d_infinity, -d_infinity);
const interval interval::universe = interval(-d_infinity, d_infinity);

/**
 * @brief Add a displacement to an interval
 * 
 * @param ival Interval to displace
 * @param displacement Value to displace by
 * @return Displaced interval
 */
__host__ __device__ interval operator+(const interval& ival, float displacement) {
    return interval(ival.min + displacement, ival.max + displacement);
}

/**
 * @brief Subtract a displacement from an interval
 * 
 * @param ival Interval to displace
 * @param displacement Value to displace by
 * @return Displaced interval
 */
__host__ __device__ interval operator-(const interval& ival, float displacement) {
    return interval(ival.min - displacement, ival.max - displacement);
}

/**
 * @brief Print a closed interval
 * 
 * @param[in] out Output stream
 * @param[in] ival interval to print
 * @return Output stream
 */
inline std::ostream& operator<<(std::ostream& out, const interval& ival) {
    return out << "[" << ival.min << ", " << ival.max << "]";
}

#endif