/**
 * @file material.cuh
 * @author Jasper Jeuken
 * @brief Defines material classes
 */
#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.cuh"
#include "texture.cuh"


/// @brief Enum for material types
enum class material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT,
    OPEN_PBR
};

class material; // Forward declaration


/**
 * @struct hit_record
 * @brief Information about a ray-object intersection
 */
struct hit_record {
    point3 p; ///< Intersection point
    vec3 normal; ///< Surface normal
    float t; ///< Ray parameter
    float u; ///< Texture coordinate u
    float v; ///< Texture coordinate v
    bool front_face; ///< Whether the ray hit the front face of the object
    int mat_idx; ///< Material index
    vec3 tangent; ///< Tangent vector
    vec3 bitangent; ///< Bitangent vector

    /**
     * @brief Set the face normal based on the ray direction and outward normal
     * 
     * @param[in] r Ray
     * @param outward_normal Outward normal
     */
    __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = vec3::dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


/**
 * @class material
 * @brief Base material
 */
class material {
public:
    material_type type; ///< Material type

    /**
     * @brief Construct a new base material object
     * 
     * @param[in] _type Material type
     * @return Constructed material
     */
    __host__ material(material_type _type) : type(_type) {}

    /// @brief Virtual destructor
    __host__ __device__ virtual ~material() {}

    /**
     * @brief Determine the scattered ray after a material interaction
     * 
     * @param[in] r_in Incoming ray
     * @param[in] rec Hit record
     * @param[out] attenuation Color attenuation
     * @param[out] scattered Outgoing ray
     * @param[in] rand_state Random state
     * @param[in] textures Textures in scene
     * @return Whether the ray was scattered 
     */
    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state, _texture** textures) const;
    
    /**
     * @brief Determine the emitted color of the material
     * 
     * @param[in] u Texture coordinate u
     * @param[in] v Texture coordinate v
     * @param[in] p Hit point
     * @param[in] textures Textures in scene
     * @return Emitted color
     */
    __device__ color emit(float u, float v, const point3& p, _texture** textures) const;
};


/**
 * @struct lambertian
 * @brief Lambertian material
 * 
 * A lambertian material scatters rays in random directions with a color attenuation.
 * This defines a diffuse material.
 * 
 * The attenuation color is determined by a texture.
 */
struct lambertian : public material {
    int tex_idx; ///< Texture index

    /**
     * @brief Construct a new lambertian object
     * 
     * @param[in] _tex_idx Texture index
     * @return Constructed lambertian material 
     */
    __host__ lambertian(int _tex_idx) : tex_idx(_tex_idx), material(material_type::LAMBERTIAN) {}
};

/**
 * @struct metal
 * @brief Metal material
 * 
 * A metal material scatters rays in a reflected direction with a color attenuation.
 * This defines a specular material.
 * 
 * The attenuation color is determined by the albedo texture.
 * The fuzziness determines the spread of the reflection.
 */
struct metal : public material {
    vec3 albedo; ///< Albedo color
    float fuzz; ///< Fuzziness

    /**
     * @brief Construct a new metal object
     * 
     * @param[in] _albedo Albedo color
     * @param[in] _fuzz Material fuzziness
     * @return Constructed metal material
     */
    __host__ metal(const vec3& _albedo, float _fuzz) : albedo(_albedo), fuzz(_fuzz), material(material_type::METAL) {}
    
    /**
     * @brief Calculate the reflectance of the material
     * 
     * Uses Schlick's approximation to calculate the reflectance of the material.
     * 
     * @param[in] cosine Cosine of the angle between the ray and the normal
     * @param[in] _albedo Albedo color
     * @return Reflectance color
     */
    __device__ inline static color reflectance(float cosine, const color& _albedo) {
        return _albedo + (color(1, 1, 1) - _albedo) * pow(1.0f - cosine, 5.0f);
    }
};


/**
 * @struct dielectric
 * @brief Dielectric material
 * 
 * A dielectric material scatters rays in a reflected or refracted direction.
 * 
 * The index of refraction determines the refractive properties of the material.
 */
struct dielectric : public material {
    float index_of_refraction; ///< Index of refraction

    /**
     * @brief Construct a new dielectric object
     * 
     * @param[in] _index_of_refraction Index of refraction
     * @return Constructed dielectric material
     */
    __host__ dielectric(float _index_of_refraction) : index_of_refraction(_index_of_refraction), material(material_type::DIELECTRIC) {}

    /**
     * @brief Calculate the reflectance of the material
     * 
     * @param[in] cosine Cosine of the angle between the ray and the normal
     * @param[in] ref_idx Refractive index
     * @return Reflectance value
     */
    __device__ inline static float reflectance(float cosine, float ref_idx) {
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
    }
};

/**
 * @struct diffuse_light
 * @brief Diffuse light material
 * 
 * A diffuse light material emits light in the direction of the normal.
 * The emitted color is determined by a texture.
 */
class diffuse_light : public material {
public:
    int tex_idx; ///< Texture index

    /**
     * @brief Construct a new diffuse light object
     * 
     * @param[in] _tex_idx Texture index
     * @return Constructed diffuse light material 
     */
    __host__ diffuse_light(int _tex_idx) : tex_idx(_tex_idx), material(material_type::DIFFUSE_LIGHT) {}
};

/**
 * @struct open_pbr
 * @brief Physically based rendering material (not implemented)
 * @todo Implement this material
 */
class open_pbr : public material {
public:
    // Base
    float base_weight = 1.0f; ///< Base weight
    color base_color = color(0.8f, 0.8f, 0.8f); ///< Base color
    float base_metalness = 0.0f; ///< Base metalness
    float base_diffuse_roughness = 0.0f; ///< Base diffuse roughness

    // Specular
    float specular_weight = 1.0f; ///< Specular weight
    color specular_color = color(1.0f, 1.0f, 1.0f); ///< Specular color
    float specular_roughness = 0.3f; ///< Specular roughness
    float specular_roughness_anisotropy = 0.0f; ///< Specular roughness anisotropy
    float specular_ior = 1.5f; ///< Specular index of refraction

    // Transmission
    float transmission_weight = 0.0f; ///< Transmission weight
    color transmission_color = color(1.0f, 1.0f, 1.0f); ///< Transmission color
    float transmission_depth = 0.0f; ///< Transmission depth
    color transmission_scatter = color(0.0f, 0.0f, 0.0f); ///< Transmission scatter
    float transmission_scatter_anisotropy = 0.0f; ///< Transmission scatter anisotropy
    float transmission_dispersion_scale = 0.0f; ///< Transmission dispersion scale
    float transmission_dispersion_abbe_number = 20.0f; ///< Transmission dispersion Abbe number

    // Subsurface
    float subsurface_weight = 0.0f; ///< Subsurface weight
    color subsurface_color = color(0.8f, 0.8f, 0.8f); ///< Subsurface color
    float subsurface_radius = 1.0f; ///< Subsurface radius
    color subsurface_radius_scale = color(1.0f, 0.5f, 0.25f); ///< Subsurface radius scale
    float subsurface_scatter_anisotropy = 0.0f; ///< Subsurface scatter anisotropy

    // Coat
    float coat_weight = 0.0f; ///< Coat weight
    color coat_color = color(1.0f, 1.0f, 1.0f); ///< Coat color
    float coat_roughness = 0.0f; ///< Coat roughness
    float coat_roughness_anisotropy = 0.0f; ///< Coat roughness anisotropy
    float coat_ior = 1.6f; ///< Coat index of refraction
    float coat_darkening = 1.0f; ///< Coat darkening

    // Fuzz
    float fuzz_weight = 0.0f; ///< Fuzz weight
    color fuzz_color = color(1.0f, 1.0f, 1.0f); ///< Fuzz color
    float fuzz_roughness = 0.5f; ///< Fuzz roughness

    // Emission
    float emission_luminance = 0.0f; ///< Emission luminance
    color emission_color = color(1.0f, 1.0f, 1.0f); ///< Emission color

    // Thin film
    float thin_film_weight = 0.0f; ///< Thin film weight
    float thin_film_thickness = 0.5f; ///< Thin film thickness
    float thin_film_ior = 1.4f; ///< Thin film index of refraction

    // Geometry
    float geometry_opacity = 1.0f; ///< Geometry opacity
    bool geometry_thin_walled = false; ///< Geometry thin walled
    vec3 geometry_normal; ///< Geometry normal
    vec3 geometry_tangent; ///< Geometry tangent
    vec3 geometry_coat_normal; ///< Geometry coat normal
    vec3 geometry_coat_tangent; ///< Geometry coat tangent
};


/**
 * @brief Scatter a lambertian material
 * 
 * @param[in] r_in Incoming ray
 * @param[in] rec Hit record
 * @param[out] attenuation Attenuation color
 * @param[out] scattered Scattered ray
 * @param[in] rand_state Random state
 * @param[in] textures Textures in scene
 * @param[in] object_data Lambertian material data
 * @return Whether the ray was scattered 
 */
__device__ bool scatter_lambertian(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state, _texture** textures, const void* object_data) {
    const lambertian* lam = reinterpret_cast<const lambertian*>(object_data);
    vec3 scatter_direction = rec.normal + vec3::random_unit_vector(rand_state);
    if (scatter_direction.near_zero()) scatter_direction = rec.normal;
    scattered = ray(rec.p + rec.normal * 0.001f, scatter_direction, r_in.time());
    attenuation = textures[lam->tex_idx]->value(rec.u, rec.v, rec.p, textures);
    return true;
}

/**
 * @brief Scatter a metal material
 * 
 * @param[in] r_in Incoming ray
 * @param[in] rec Hit record
 * @param[out] attenuation Attenuation color
 * @param[out] scattered Scattered ray
 * @param[in] rand_state Random state
 * @param[in] textures Textures in scene
 * @param[in] object_data Metal material data
 * @return Whether the ray was scattered 
 */
__device__ bool scatter_metal(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state, _texture** textures, const void* object_data) {
    const metal* met = reinterpret_cast<const metal*>(object_data);
    // Calculate new ray based on reflection and fuzziness
    vec3 reflected = vec3::reflect(vec3::unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + met->fuzz * vec3::random_in_unit_sphere(rand_state), r_in.time());

    // Calculate attenuation using Schlick's approximation
    vec3 unit_direction = vec3::unit_vector(r_in.direction());
    float cos_theta = min(vec3::dot(-unit_direction, rec.normal), 1.0f);
    attenuation = met->reflectance(cos_theta, met->albedo);
    return vec3::dot(scattered.direction(), rec.normal) > 0.0f;
}

/**
 * @brief Scatter a dielectric material (reflected or refracted)
 * 
 * @param[in] r_in Incoming ray
 * @param[in] rec Hit record
 * @param[out] attenuation Attenuation color
 * @param[out] scattered Scattered ray
 * @param[in] rand_state Random state
 * @param[in] textures Textures in scene
 * @param[in] object_data Dielectric material data
 * @return Whether the ray was scattered 
 */
__device__ bool scatter_dielectric(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state, _texture** textures, const void* object_data) {
    const dielectric* die = reinterpret_cast<const dielectric*>(object_data);
    attenuation = color(1.0, 1.0, 1.0);
    float refraction_ratio = rec.front_face ? (1.0f / die->index_of_refraction) : die->index_of_refraction;

    // Calculate reflection and refraction
    vec3 unit_direction = vec3::unit_vector(r_in.direction());
    float cos_theta = minf(vec3::dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

    // Depending on the refraction ratio and the angle, the ray will be reflected or refracted
    vec3 direction;
    if (cannot_refract || die->reflectance(cos_theta, refraction_ratio) > curand_uniform(rand_state)) {
        direction = vec3::reflect(unit_direction, rec.normal);
    } else {
        direction = vec3::refract(unit_direction, rec.normal, refraction_ratio);
    }
    scattered = ray(rec.p, direction, r_in.time());
    return true;
}

// Implementation of material scatter
__device__ bool material::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state, _texture** textures) const {
    switch (type) {
    case material_type::LAMBERTIAN:
        return scatter_lambertian(r_in, rec, attenuation, scattered, rand_state, textures, this);
    case material_type::METAL:
        return scatter_metal(r_in, rec, attenuation, scattered, rand_state, textures, this);
    case material_type::DIELECTRIC:
        return scatter_dielectric(r_in, rec, attenuation, scattered, rand_state, textures, this);
    case material_type::DIFFUSE_LIGHT:
        return false;
    default:
        return false;
    }
}

// Implementation of material emit
__device__ color material::emit(float u, float v, const point3& p, _texture** textures) const {
    switch (type) {
    case material_type::DIFFUSE_LIGHT:
        return textures[reinterpret_cast<const diffuse_light*>(this)->tex_idx]->value(u, v, p, textures);
    }
    return color(0, 0, 0);
}

/**
 * @brief Copy materials to device memory
 * 
 * @param[in] h_materials Host materials
 * @return Device materials
 */
material** allocate_materials(const std::vector<material*>& h_materials) {
    material** d_materials;
    size_t material_count = h_materials.size();
    checkCudaErrors(cudaMalloc(&d_materials, material_count * sizeof(material*)));

    std::vector<material*> temp_vector(material_count);
    for (size_t i = 0; i < material_count; i++) {
        material* d_mat = nullptr;
        if (auto lam = dynamic_cast<lambertian*>(h_materials[i])) {
            d_mat = copy_object(*lam);
        } else if (auto met = dynamic_cast<metal*>(h_materials[i])) {
            d_mat = copy_object(*met);
        } else if (auto die = dynamic_cast<dielectric*>(h_materials[i])) {
            d_mat = copy_object(*die);
        } else if (auto dif = dynamic_cast<diffuse_light*>(h_materials[i])) {
            d_mat = copy_object(*dif);
        }
        temp_vector[i] = d_mat;
    }

    checkCudaErrors(cudaMemcpy(d_materials, temp_vector.data(), material_count * sizeof(material*), cudaMemcpyHostToDevice));
    return d_materials;
}

/**
 * @brief Print material
 * 
 * @param[in] out Output stream
 * @param[in] mat Material to print
 * @return Output stream
 */
inline std::ostream& operator<<(std::ostream& out, const material* mat) {
    if (mat->type == material_type::LAMBERTIAN) {
        lambertian* lam = (lambertian*)mat;
        return out << "<lambertian tex_idx=" << lam->tex_idx << ">";
    } else if (mat->type == material_type::METAL) {
        metal* met = (metal*)mat;
        return out << "<metal albedo=(" << met->albedo << ") fuzz=" << met->fuzz << ">";
    } else if (mat->type == material_type::DIELECTRIC) {
        dielectric* die = (dielectric*)mat;
        return out << "<dielectric index_of_refraction=" << die->index_of_refraction << ">";
    } else if (mat->type == material_type::DIFFUSE_LIGHT) {
        diffuse_light* dif = (diffuse_light*)mat;
        return out << "<diffuse_light tex_idx=" << dif->tex_idx << ">";
    } else {
        return out << "<Unknown material>";
    }
}

#endif