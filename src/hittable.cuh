/**
 * @file hittable.cuh
 * @author Jasper Jeuken
 * @brief Defines hittable objects
 */
#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.cuh"
#include "material.cuh"
#include "interval.cuh"
#include "vec2.cuh"
#include "mat3.cuh"

/// @brief Enum for object types
enum class object_type {
    BVH,
    SPHERE,
    TRI
};


/**
 * @class hittable
 * @brief Base hittable object
 */
class hittable {
public:
    int mat_idx; ///< Material index
    object_type type; ///< Object type
    aabb bbox; ///< Bounding box
    point3 anchor_point = point3(0, 0, 0); ///< Anchor point for transformations
    vec3 translation = vec3(0, 0, 0); ///< Translation vector
    mat3 rotation = mat3::identity(); ///< Rotation matrix
    mat3 rotation_inv = mat3::identity(); ///< Inverse rotation matrix
    bool visible; ///< Visibility
    int normal_idx = -1; ///< Normal map index

    /**
     * @brief Construct a new hittable object
     * 
     * @param[in] _type Object type
     * @param[in] _mat_idx Material index
     * @param[in] _visible Visibility
     * @return Constructed hittable object
     */
    __host__ hittable(object_type _type, int _mat_idx, bool _visible = true) : type(_type), mat_idx(_mat_idx), bbox(), visible(_visible) {}

    __host__ virtual ~hittable() {} ///< Virtual destructor

    /**
     * @brief Determine if a ray intersects the object
     * 
     * @param[in] r Ray to test
     * @param[in] ray_t Interval on ray
     * @param[out] rec Hit record
     * @param[in] rand_state Random state
     * @param[in] textures Textures in scene
     * @return Whether the ray intersects the object
     */
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* rand_state, _texture** textures) const;

    /**
     * @brief Get the bounding box of the object
     * 
     * @return Bounding box of the object
     */
    __host__ __device__ aabb bounding_box() const {
        return bbox;
    };

    /**
     * @brief Translate the object
     * 
     * @param[in] offset Translation vector
     */
    __host__ void translate(const vec3& offset) {
        translation += offset;
        bbox = aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
        anchor_point = anchor_point + offset;
    }

    /**
     * @brief Rotate the object
     * 
     * @param[in] angles Rotation angles in degrees (x, y, z)
     * 
     * @note Rotations are applied in the order z, y, x
     */
    __host__ void rotate(const vec3& angles, const vec3& anchor = vec3(0, 0, 0)) {
        anchor_point = anchor;

        // Calculate rotation matrix
        vec3 rad_angles = vec3(degrees_to_radians(angles.x()), degrees_to_radians(angles.y()), degrees_to_radians(angles.z()));
        rotation = mat3::from_euler(rad_angles);
        rotation_inv = rotation.inverse();

        // Get corners of current bounding box
        vec3 corners[8] = {
            vec3(bbox.x.min, bbox.y.min, bbox.z.min),
            vec3(bbox.x.max, bbox.y.min, bbox.z.min),
            vec3(bbox.x.min, bbox.y.max, bbox.z.min),
            vec3(bbox.x.min, bbox.y.min, bbox.z.max),
            vec3(bbox.x.max, bbox.y.max, bbox.z.min),
            vec3(bbox.x.min, bbox.y.max, bbox.z.max),
            vec3(bbox.x.max, bbox.y.min, bbox.z.max),
            vec3(bbox.x.max, bbox.y.max, bbox.z.max)
        };

        // Rotate bounding box
        vec3 rotated_corners[8];
        for (int i = 0; i < 8; i++) {
            vec3 relative = corners[i] - anchor_point;
            rotated_corners[i] = rotation * relative + anchor_point;
        }

        // Calculate new bounding box
        vec3 new_min = rotated_corners[0];
        vec3 new_max = rotated_corners[0];
        for (int i = 0; i < 8; i++) {
            for (int c = 0; c < 3; c++) {
                new_min[c] = fmin(new_min[c], rotated_corners[i][c]);
                new_max[c] = fmax(new_max[c], rotated_corners[i][c]);
            }
        }
        bbox = aabb(new_min, new_max);
    }
};


/**
 * @class sphere
 * @brief Sphere object
 */
struct sphere : public hittable {
    ray center; ///< Center of the sphere
    float radius; ///< Radius of the sphere

    /**
     * @brief Construct a new sphere object
     * 
     * @param[in] _center Sphere center
     * @param[in] _radius Sphere radius
     * @param[in] _mat_idx Material index
     * @param[in] _visible Visibility
     * @return Constructed sphere object
     */
    __host__ sphere(const point3& _center, float _radius, int _mat_idx, bool _visible = true) : center(_center, vec3(0, 0, 0)), radius(_radius), hittable(object_type::SPHERE, _mat_idx, _visible) {
        point3 rvec = point3(_radius, _radius, _radius);
        bbox = aabb(_center - rvec, _center + rvec);
        anchor_point = _center;
    }

    /**
     * @brief Construct a new moving sphere object
     * 
     * @param[in] _center1 Sphere center at time 0
     * @param[in] _center2 Sphere center at time 1
     * @param[in] _radius Sphere radius
     * @param[in] _mat_idx Material index
     * @param[in] _visible Visibility
     * @return Constructed moving sphere object
     */
    __host__ sphere(const point3& _center1, const point3& _center2, float _radius, int _mat_idx, bool _visible = true) : center(_center1, _center2 - _center1), radius(_radius), hittable(object_type::SPHERE, _mat_idx, _visible) {
        point3 rvec = point3(_radius, _radius, _radius);
        aabb bbox1(center.at(0) - rvec, center.at(0) + rvec);
        aabb bbox2(center.at(1) - rvec, center.at(1) + rvec);
        bbox = aabb(bbox1, bbox2);
        anchor_point = (_center1 + _center2) / 2;
    }

    /**
     * @brief Get the UV coordinates of a point on the sphere
     * 
     * @param[in] p Point on the sphere
     * @param[out] u U coordinate
     * @param[out] v V coordinate
     */
    __device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
        float theta = acosf(-p.y());
        float phi = atan2f(-p.z(), p.x()) + d_pi;
        u = phi / (2 * d_pi);
        v = theta / d_pi;
    }

    /**
     * @brief Get the tangent and bitangent vectors at a hit point
     * 
     * @param[out] rec Hit record
     */
    __device__ void get_tangent_bitangent(hit_record& rec) const {
        vec3 local_p = rec.p - center.at(rec.t);

        float phi = atan2f(-local_p.z(), local_p.x()) + d_pi;
        rec.tangent = vec3::unit_vector(vec3(-radius * sinf(phi), 0, radius * cosf(phi)));
        rec.bitangent = vec3::unit_vector(vec3::cross(rec.normal, rec.tangent));
    }
};

/**
 * @class tri
 * @brief Triangle object
 */
class tri : public hittable {
public:

    /// @brief Default constructor
    __host__ tri() : hittable(object_type::TRI, -1, false) {}

    /**
     * @brief Construct a new triangle object
     * 
     * @param[in] v0 First vertex
     * @param[in] v1 Second vertex
     * @param[in] v2 Third vertex
     * @param[in] _normal0 First vertex normal
     * @param[in] _normal1 Second vertex normal (only used if shade_smooth)
     * @param[in] _normal2 Third vertex normal (only used if shade_smooth)
     * @param[in] _uv0 First vertex UV
     * @param[in] _uv1 Second vertex UV
     * @param[in] _uv2 Third vertex UV
     * @param[in] _mat_idx Material index
     * @param[in] _shade_smooth Whether to shade smooth
     * @param[in] _visible Visibility
     * @return Constructed triangle object
     */
    __host__ tri(vec3 v0, vec3 v1, vec3 v2, vec3 _normal0, vec3 _normal1, vec3 _normal2, vec2 _uv0, vec2 _uv1, vec2 _uv2, int _mat_idx, bool _shade_smooth = false, bool _visible = true) 
        : point(v0), edge1(v1 - v0), edge2(v2 - v0), normal0(_normal0), normal1(_normal1), normal2(_normal2), uv0(_uv0), uv1(_uv1), uv2(_uv2), shade_smooth(_shade_smooth), hittable(object_type::TRI, _mat_idx, _visible) {
        
        // Calculate bounding box
        bbox = aabb(v0, v1, v2);

        // Use flat normal if not shaded smooth
        if (!shade_smooth) {
            normal0 = vec3::unit_vector(vec3::cross(edge1, edge2));
            normal1 = normal0; // arbitrary
            normal2 = normal0; // arbitrary
        }

        // Precompute tangent and bitangent
        vec2 delta_uv1 = uv1 - uv0;
        vec2 delta_uv2 = uv2 - uv0;
        float f = 1.0f / (delta_uv1.x() * delta_uv2.y() - delta_uv2.x() * delta_uv1.y());
        tangent = f * (delta_uv2.y() * edge1 - delta_uv1.y() * edge2);
        bitangent = f * (-delta_uv2.x() * edge1 + delta_uv1.x() * edge2);
    }

public:
    vec3 point; ///< First vertex
    vec3 edge1; ///< Second vertex
    vec3 edge2; ///< Third vertex
    vec3 normal0; ///< First vertex normal
    vec3 normal1; ///< Second vertex normal (only used if shade_smooth)
    vec3 normal2; ///< Third vertex normal (only used if shade_smooth)
    vec2 uv0; ///< First vertex UV
    vec2 uv1; ///< Second vertex UV
    vec2 uv2; ///< Third vertex UV
    vec3 tangent; ///< Tangent vector
    vec3 bitangent; ///< Bitangent vector
    bool shade_smooth; ///< Whether to shade smooth
};

/**
 * @brief Determine if a ray intersects a sphere
 * 
 * @param[in] r Ray to test
 * @param[in] ray_t Interval on ray
 * @param[out] rec Hit record
 * @param[in] object_data Sphere data
 * @return Whether the ray intersects the sphere 
 */
__device__ __forceinline__ bool hit_sphere(const ray& r, interval ray_t, hit_record& rec, const void* object_data) {
    const sphere* sph = reinterpret_cast<const sphere*>(object_data);
    
    //Calculate if hit using quadratic formula
    point3 current_center = sph->center.at(r.time());
    vec3 oc = r.origin() - current_center;
    float a = r.direction().length_squared();
    float h = vec3::dot(oc, r.direction());
    float c = oc.length_squared() - sph->radius * sph->radius;
    float discriminant = h * h - a * c;
    if (discriminant < 0) return false;

    // Check if either root is within specified range
    float sqrt_discriminant = sqrt(discriminant);
    float root = (- h - sqrt_discriminant) / a;
    if (!ray_t.surrounds(root)) {
        root = (- h + sqrt_discriminant) / a;
        if (!ray_t.surrounds(root)) return false;
    }

    // Store hit information
    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - current_center) / sph->radius;
    rec.set_face_normal(r, outward_normal);
    sph->get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_idx = sph->mat_idx;
    sph->get_tangent_bitangent(rec);
    return true;
}

/**
 * @brief Determine if a ray intersects a triangle
 * 
 * Uses the Möller-Trumbore algorithm to determine if a ray intersects a triangle.
 * 
 * @param[in] r Ray to test
 * @param[in] ray_t Interval on ray
 * @param[out] rec Hit record
 * @param[in] object_data Triangle data
 * @return Whether the ray intersects the triangle 
 */
__device__ __forceinline__ bool hit_tri(const ray& r, interval ray_t, hit_record& rec, const void* object_data, float epsilon = 1e-8f) {
    const tri* tr = reinterpret_cast<const tri*>(object_data);

    // Calculate determinant
    vec3 pvec = vec3::cross(r.direction(), tr->edge2);
    float det = vec3::dot(tr->edge1, pvec);
    if (fabsf(det) < epsilon) return false;

    // Calculate u
    float inv_det = 1.0f / det;
    vec3 tvec = r.origin() - tr->point;
    float u = vec3::dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    // Calculate v
    vec3 qvec = vec3::cross(tvec, tr->edge1);
    float v = vec3::dot(r.direction(), qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    // Calculate t
    float t = vec3::dot(tr->edge2, qvec) * inv_det;
    if (!ray_t.contains(t)) return false;

    // Update hit record
    rec.t = t;
    rec.p = r.at(t);
    rec.mat_idx = tr->mat_idx;
    rec.tangent = tr->tangent;
    rec.bitangent = tr->bitangent;

    // Calculate normal
    float w = 1.0f - u - v;
    if (tr->shade_smooth) {
        rec.normal = vec3::unit_vector(w * tr->normal0 + u * tr->normal1 + v * tr->normal2);
    } else {
        rec.normal = tr->normal0;
    }
    rec.set_face_normal(r, rec.normal);

    // Calculate UV coordinates
    vec2 uv = w * tr->uv0 + u * tr->uv1 + v * tr->uv2;
    rec.u = uv.x();
    rec.v = uv.y();

    return true;
}

/**
 * @brief Apply a normal map to a hit record
 * 
 * @param[in,out] rec Hit record
 * @param[in] texture Normal map texture
 */
__device__ void apply_normal_map(hit_record& rec, _texture* texture) {
    // Get normal map pixel
    const image* normal_map = reinterpret_cast<const image*>(texture);
    vec3 pixel = vec3(0, 0, 1);
    if (normal_map->channels == 1) {
        float value = tex2D<float>(normal_map->tex, rec.u, rec.v) * 2.0f - 1.0f;
        pixel = vec3(value, value, value);
    } else if (normal_map->channels == 4) {
        float4 value = tex2D<float4>(normal_map->tex, rec.u, rec.v);
        pixel = vec3(value.x * 2.0f - 1.0f, value.y * 2.0f - 1.0f, value.z * 2.0f - 1.0f);
    }
    
    // Apply TBN calculation
    vec3 world_normal = vec3(
        rec.tangent[0] * pixel[0] + rec.bitangent[0] * pixel[1] + rec.normal[0] * pixel[2],
        rec.tangent[1] * pixel[0] + rec.bitangent[1] * pixel[1] + rec.normal[1] * pixel[2],
        rec.tangent[2] * pixel[0] + rec.bitangent[2] * pixel[1] + rec.normal[2] * pixel[2]
    );
    rec.normal = vec3::unit_vector(world_normal);
}

/**
 * @brief Determine if a ray intersects an object
 * 
 * @param[in] r Ray to test
 * @param[in] ray_t Interval on ray
 * @param[out] rec Hit record
 * @param[in] rand_state Random state
 * @param[in] object_data Object data
 * @return Whether the ray intersects the object 
 */
__device__ bool object_hit(const ray& r, interval ray_t, hit_record& rec, curandState* rand_state, const void* object_data) {
    const hittable* obj = reinterpret_cast<const hittable*>(object_data);

    bool hit;
    switch (obj->type) {
    case object_type::SPHERE:
        hit = hit_sphere(r, ray_t, rec, object_data);
        break;
    case object_type::TRI:
        hit = hit_tri(r, ray_t, rec, object_data);
        break;
    default:
        hit = false;
    }
    return hit;
}


/**
 * @brief Determine if a ray intersects an object with transformations
 * 
 * @param[in] r Ray to test
 * @param[in] ray_t Interval on ray
 * @param[out] rec Hit record
 * @param[in] rand_state Random state
 * @param[in] textures Textures in scene
 * @return Whether the ray intersects the object
 */
__device__ bool hittable::hit(const ray& r, interval ray_t, hit_record& rec, curandState* rand_state, _texture** textures) const {

    // Apply transformations
    ray transformed_r = ray(r.origin() - translation, r.direction(), r.time());
    transformed_r = ray(rotation_inv * (transformed_r.origin() - anchor_point) + anchor_point, rotation_inv * transformed_r.direction(), transformed_r.time());

    // Check if object is hit
    bool hit = object_hit(transformed_r, ray_t, rec, rand_state, this);
    if (!hit) return hit;

    // Revert transformations
    vec3 rotated_p = rotation * (rec.p - anchor_point) + anchor_point;
    rec.p = rotated_p + translation;
    rec.normal = rotation * rec.normal;

    // Apply maps
    if (normal_idx >= 0) apply_normal_map(rec, textures[normal_idx]);
    return hit;
}

/**
 * @brief Copy objects to device memory
 * 
 * @param[in] h_objects Host objects
 * @return Device objects
 */
hittable** allocate_objects(const std::vector<hittable*>& h_objects) {
    hittable** d_objects;
    size_t object_count = h_objects.size();
    checkCudaErrors(cudaMalloc(&d_objects, object_count * sizeof(hittable*)));
    
    // Allocate objects on device
    std::vector<hittable*> temp_vector(object_count);
    for (size_t i = 0; i < object_count; i++) {
        hittable* d_obj = nullptr;
        if (auto sph = dynamic_cast<sphere*>(h_objects[i])) {
            d_obj = copy_object(*sph);
        } else if (auto t = dynamic_cast<tri*>(h_objects[i])) {
            d_obj = copy_object(*t);
        }
        temp_vector[i] = d_obj;
    }
    checkCudaErrors(cudaMemcpy(d_objects, temp_vector.data(), object_count * sizeof(hittable*), cudaMemcpyHostToDevice));

    return d_objects;
}

/**
 * @brief Print hittable object
 * 
 * @param[in] out Output stream
 * @param[in] obj Object to print
 * @return Output stream
 */
inline std::ostream& operator<<(std::ostream& out, const hittable* obj) {
    if (obj->type == object_type::SPHERE) {
        sphere* sph = (sphere*)obj;
        return out << "<sphere center1=(" << sph->center.origin() << ") center2=(" << sph->center.direction() << ") radius=" << sph->radius << ">";
    } else if (obj->type == object_type::TRI) {
        tri* tr = (tri*)obj;
        return out << "<triangle v0=" << tr->point << " v1=" << tr->point + tr->edge1 << " v2=" << tr->point + tr->edge2 << ">";
    } else {
        return out << "<Unknown object>";
    }
}

#endif