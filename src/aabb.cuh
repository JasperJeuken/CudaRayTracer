/**
 * @file aabb.cuh
 * @author Jasper Jeuken
 * @brief Defines an axis-aligned bounding box (AABB) in 3D space
 */
#ifndef AABB_H
#define AABB_H

#include "interval.cuh"
#include "ray.cuh"


/**
 * @class aabb
 * @brief Axis-aligned bounding box (AABB)
 * 
 * This class defines an AABB in 3D space using three interval ranges (x, y, z).
 * It provides default empty and universe bounding boxes.
 */
class aabb {
public:

    /**
     * @brief Default constructor
     * 
     * @return Empty AABB
     */
    __host__ __device__ aabb() : x(), y(), z() {}

    /**
     * @brief Construct a new AABB from x,y,z intervals
     * 
     * @param[in] _x X-interval
     * @param[in] _y Y-interval
     * @param[in] _z Z-interval
     * @return AABB that bounds the x,y,z intervals 
     */
    __host__ __device__ aabb(const interval& _x, const interval& _y, const interval& _z) : x(_x), y(_y), z(_z) {
        pad_to_minimums();
    }

    /**
     * @brief Construct a new AABB from two points
     * 
     * @param[in] a First point
     * @param[in] b Second point
     * @return AABB that bounds the two points
     */
    __host__ __device__ aabb(const point3& a, const point3& b) {
        x = (a.x() <= b.x()) ? interval(a.x(), b.x()) : interval(b.x(), a.x());
        y = (a.y() <= b.y()) ? interval(a.y(), b.y()) : interval(b.y(), a.y());
        z = (a.z() <= b.z()) ? interval(a.z(), b.z()) : interval(b.z(), a.z());
        pad_to_minimums();
    }

    /**
     * @brief Construct a new AABB spanning three points
     * 
     * @param[in] a First point
     * @param[in] b Second point
     * @param[in] c Third point
     * @return AABB that bounds the three points
     */
    __host__ __device__ aabb(const point3& a, const point3& b, const point3& c) {
        float x_min = fminf(a.x(), fminf(b.x(), c.x()));
        float x_max = fmaxf(a.x(), fmaxf(b.x(), c.x()));
        x = interval(x_min, x_max);
        
        float y_min = fminf(a.y(), fminf(b.y(), c.y()));
        float y_max = fmaxf(a.y(), fmaxf(b.y(), c.y()));
        y = interval(y_min, y_max);
        
        float z_min = fminf(a.z(), fminf(b.z(), c.z()));
        float z_max = fmaxf(a.z(), fmaxf(b.z(), c.z()));
        z = interval(z_min, z_max);

        pad_to_minimums();
    }

    /**
     * @brief Construct a new AABB spanning two AABBs
     * 
     * @param[in] a First AABB
     * @param[in] b Second AABB
     * @return AABB that bounds the two AABBs
     */
    __host__ __device__ aabb(const aabb& a, const aabb& b) {
        x = interval(a.x, b.x);
        y = interval(a.y, b.y);
        z = interval(a.z, b.z);
        pad_to_minimums();
    }

    /**
     * @brief Construct a new AABB from min and max values
     * 
     * @param[in] min_bounds Min values
     * @param[in] max_bounds Max values
     * @return AABB that bounds the min and max values
     */
    __host__ __device__ aabb(const float3 min_bounds, const float3 max_bounds) {
        // Construct from min and max bounds
        x = interval(min_bounds.x, max_bounds.x);
        y = interval(min_bounds.y, max_bounds.y);
        z = interval(min_bounds.z, max_bounds.z);
        pad_to_minimums();
    }

    /**
     * @brief Index operator to select x,y, or z interval
     * 
     * @param[in] i Interval to select (`0=x`, `1=y`, `2=z`)
     * @return Reference to the selected interval
     */
    __host__ __device__ inline const interval& operator[](int i) const {
        if (i == 0) {
            return x;
        } else if (i == 1) {
            return y;
        } else {
            return z;
        }
    }

    /**
     * @brief Determine if a ray intersects the AABB
     * 
     * @param[in] r Ray to test
     * @param[in] ray_t Interval on the ray
     * @return Whether the ray intersects the AABB
     */
    __device__ bool hit(const ray& r, interval ray_t) const {
        const point3& ray_orig = r.origin();
        const vec3& ray_dir = r.direction();

        float t_min = ray_t.min;
        float t_max = ray_t.max;

        for (int i = 0; i < 3; i++) {
            float inv_d = (fabsf(ray_dir[i]) > 1e-12f) ? (1.0f / ray_dir[i]) : 1e8f;
            float t0 = ((*this)[i].min - ray_orig[i]) * inv_d;
            float t1 = ((*this)[i].max - ray_orig[i]) * inv_d;

            if (inv_d < 0.0f) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            t_min = fmaxf(t_min, t0);
            t_max = fminf(t_max, t1);

            if (t_max < t_min) return false;
        }
        return true;
    }

    /**
     * @brief Find the longest axis of the AABB
     * 
     * @return Index of the longest axis (`0=x`, `1=y`, `2=z`)
     */
    __host__ int longest_axis() const {
        if (x.size() > y.size()) return x.size() > z.size() ? 0 : 2;
        return y.size() > z.size() ? 1 : 2;
    }

    /**
     * @brief Calculate the center of the AABB
     * 
     * @return Center point of the AABB
     */
    __host__ __device__ point3 center() const {
        return point3(x.center(), y.center(), z.center());
    }

    /**
     * @brief Calculate the surface area of the AABB
     * 
     * @return Surface area of the AABB 
     */
    __host__ __device__ float surface_area() const {
        return 2.0f * (x.size() * y.size() + x.size() * z.size() + y.size() * z.size());
    }

    /**
     * @brief Find the minimum point of the AABB
     * 
     * @return Minimum point of the AABB
     */
    __host__ __device__ point3 min() const {
        return point3(x.min, y.min, z.min);
    }

    /**
     * @brief Find the maximum point of the AABB
     * 
     * @return Maximum point of the AABB
     */
    __host__ __device__ point3 max() const {
        return point3(x.max, y.max, z.max);
    }

    static const aabb empty; ///< Predefined empty AABB
    static const aabb universe; ///< Predefined universe AABB

    interval x; ///< X-axis interval
    interval y; ///< Y-axis interval
    interval z; ///< Z-axis interval

private:
/**
 * @brief Pad intervals to minimum size
 * 
 * To avoid near-zero intervals, this function expands the intervals to a minimum size.
 * 
 * @param[in] delta Minimum size
 */
    __host__ __device__ void pad_to_minimums(float delta = 0.001f) {
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};

/// @brief Definition of an empty AABB
const aabb aabb::empty = aabb(interval::empty, interval::empty, interval::empty);

/// @brief Definition of a universe AABB
const aabb aabb::universe = aabb(interval::universe, interval::universe, interval::universe);

/**
 * @brief Add an offset to an AABB
 * 
 * @param[in] a AABB to offset
 * @param[in] offset Offset to add
 * @return AABB with offset added
 */
__host__ aabb operator+(const aabb& a, const vec3& offset) {
    return aabb(a.x + offset.x(), a.y + offset.y(), a.z + offset.z());
}

/**
 * @brief Print an AABB
 * 
 * @param[in] out Output stream
 * @param[in] bbox AABB to print
 * @return Output stream
 */
inline std::ostream& operator<<(std::ostream& out, const aabb& bbox) {
    return out << "bbox(" << bbox.x << " " << bbox.y << " " << bbox.z << ")";
}


#endif
