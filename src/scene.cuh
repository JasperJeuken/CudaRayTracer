/**
 * @file scene.cuh
 * @author Jasper Jeuken
 * @brief Defines a scene class and related functions
 */
#ifndef SCENE_H
#define SCENE_H

#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <unordered_map>

#include "hittable.cuh"
#include "camera.cuh"
#include "interval.cuh"
#include "bvh.cuh"
#include "texture.cuh"
#include "objects.cuh"


/**
 * @struct device_scene
 * @brief Information about a scene (device only)
 */
struct device_scene {
    hittable** d_objects; ///< Objects in the scene
    material** d_materials; ///< Materials in the scene
    _texture** d_textures; ///< Textures in the scene
    bvh4_tree* d_bvh4; ///< BVH4 tree
    int object_count; ///< Number of objects
    int material_count; ///< Number of materials
    int texture_count; ///< Number of textures
    int background_index; ///< Index of the background texture
    int bvh4_count; ///< Number of BVH4 nodes
    camera* d_camera; ///< Camera in the scene
};

/**
 * @class scene
 * @brief Information about a scene (host only)
 * 
 * Used to store information about a scene on the host before copying it to the device.
 */
class scene {
public:
    camera_information camera_info; ///< Scene camera information
    render_information render_info; ///< Scene render information
    device_scene* d_scene; ///< Device scene

    /**
     * @brief Construct a new scene
     * 
     * @param[in] h_objects Objects in the scene
     * @param[in] h_materials Materials in the scene
     * @param[in] h_textures Textures in the scene
     * @param[in] _camera Camera in the scene
     * @param[in] background_idx Index of the background texture
     * @return Constructed scene
     */
    __host__ scene(std::vector<hittable*> h_objects, std::vector<material*> h_materials, std::vector<_texture*> h_textures, camera* _camera, int background_idx) {
        d_scene = new device_scene();
        
        // Store the background as a texture
        d_scene->background_index = background_idx;
        
        // Store the number of objects, materials, and textures
        d_scene->object_count = h_objects.size();
        d_scene->material_count = h_materials.size();
        d_scene->texture_count = h_textures.size();

        // Create BVH
        bvh4_tree tree = create_bvh4(h_objects);
        d_scene->bvh4_count = tree.node_count;
        d_scene->d_bvh4 = allocate_bvh4(tree);

        // Allocate objects, materials, and textures on device
        d_scene->d_textures = allocate_textures(h_textures);
        d_scene->d_materials = allocate_materials(h_materials);
        d_scene->d_objects = allocate_objects(h_objects);

        // Allocate camera on device
        camera_info = _camera->cam_info;
        render_info = _camera->render_info;
        d_scene->d_camera = allocate_camera(_camera);

        // Allocate and copy the temporary device scene to the device
        d_scene = copy_object(*d_scene);
    }

    /// @brief Free the scene from host and device memory
    __host__ ~scene() {
        if (!d_scene) {
            return;
        }
        
        // Take scene from device
        d_scene = copy_object_to_host(*d_scene);

        // Free textures
        if (d_scene->d_textures) {
            free_textures(d_scene->d_textures, d_scene->texture_count);
            // free_pointer_array(d_scene->d_textures, d_scene->texture_count);
            d_scene->d_textures = nullptr;
        }

        // Free materials
        if (d_scene->d_materials) {
            free_pointer_array(d_scene->d_materials, d_scene->material_count);
            d_scene->d_materials = nullptr;
        }

        // Free objects
        if (d_scene->d_objects) {
            free_pointer_array(d_scene->d_objects, d_scene->object_count);
            d_scene->d_objects = nullptr;
        }

        // Free camera
        if (d_scene->d_camera) {
            free_object(d_scene->d_camera);
            d_scene->d_camera = nullptr;
        }

        // Free BVH
        if (d_scene->d_bvh4) {
            free_bvh4(d_scene->d_bvh4, d_scene->bvh4_count);
            // free_object(d_scene->d_bvh4);
            d_scene->d_bvh4 = nullptr;
        }

        delete d_scene;
    }

    /**
     * @brief Get the device scene object
     * 
     * @return Device scene
     */
    __host__ device_scene* get_device_scene() {
        return d_scene;
    }
};

/**
 * @brief Parse the camera settings YAML
 * 
 * @param[in] node Camera YAML node
 * @param[in] cam_type Camera type (`perspective` or `orthographic`)
 * @return Parsed camera information
 */
camera_information parse_camera_settings(const YAML::Node& node, std::string cam_type) {
    camera_information cam_info;

    // Get shared camera settings
    cam_info.from = parse_required_vec3(node, "from", "Camera 'from' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
    cam_info.to = parse_required_vec3(node, "to", "Camera 'to' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
    cam_info.up = parse_optional_vec3(node, "up", vec3(0, 1, 0));

    if (cam_type == "perspective") {
        // Get perspective camera settings
        cam_info.vfov = parse_required<float>(node, "vfov", "Camera 'vfov' float missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        cam_info.defocus_angle = parse_optional<float>(node, "defocus_angle", 0.0f);
        cam_info.focus_dist = parse_optional<float>(node, "focus_dist", 10.0f);

    } else if (cam_type == "orthographic") {
        // Get orthographic camera settings
        cam_info.width = parse_required<float>(node, "width", "Camera 'width' float missing or invalid (line " + std::to_string(node.Mark().line) + ")");

    } else {
        error_with_message("Unknown camera type '" + cam_type + "', must be either 'perspective' or 'orthographic' (line " + std::to_string(node.Mark().line) + ")");
    }

    return cam_info;
}

/**
 * @brief Parse the render settings YAML
 * 
 * @param[in] node Render settings YAML node
 * @return Parsed render information
 */
render_information parse_render_information(const YAML::Node& node) {
    render_information render_info;

    // Get required values
    render_info.image_width = parse_required<int>(node, "width", "Render 'width' integer missing or invalid (line " + std::to_string(node.Mark().line) + ")");
    render_info.image_height = parse_required<int>(node, "height", "Render 'height' integer missing or invalid (line " + std::to_string(node.Mark().line) + ")");
    render_info.samples_per_pixel = parse_required<int>(node, "samples", "Render 'samples' integer missing or invalid (line " + std::to_string(node.Mark().line) + ")");

    // Get optional values
    render_info.max_bounces = parse_optional<int>(node, "max_bounces", 50);
    render_info.gamma = parse_optional<float>(node, "gamma", 2.0f);

    return render_info;
}

/**
 * @brief Parse render and camera settings YAML
 * 
 * @param config Root YAML node
 * @return Camera based on settings
 */
camera* parse_settings(const YAML::Node& config) {
    // Get camera node
    const YAML::Node& cam_node = config["camera"];
    if (!cam_node || !cam_node.IsMap()) {
        error_with_message("Camera settings missing or invalid");
    }

    // Get render node
    const YAML::Node& render_node = config["render"];
    if (!render_node || !render_node.IsMap()) {
        error_with_message("Render settings missing or invalid");
    }

    // Get camera type
    std::string type = parse_required<std::string>(cam_node, "type", "Camera type missing or invalid (line " + std::to_string(cam_node.Mark().line) + ")");

    // Parse settings
    camera_information cam_info = parse_camera_settings(cam_node, type);
    render_information render_info = parse_render_information(render_node);

    // Create camera object
    if (type == "perspective") {
        return new perspective_camera(cam_info, render_info);
    } else if (type == "orthographic") {
        return new orthographic_camera(cam_info, render_info);
    }
    throw std::runtime_error("Unknown camera type");
}

/**
 * @brief Parse a texture YAML node
 * 
 * @param[in] node Texture YAML node
 * @param[in,out] textures Textures in the scene
 * @param[in,out] texture_map Map of named textures
 * @return Index of parsed texture
 */
int parse_texture(const YAML::Node& node, std::vector<_texture*>& textures, std::unordered_map<std::string, int>& texture_map) {
    // Parse scalar texture
    if (node.IsScalar()) {
        std::string name = node.as<std::string>();
        if (!texture_map.count(name)) {
            error_with_message("Texture '" + name + "' not found (line " + std::to_string(node.Mark().line) + ")");
        }
        return texture_map[name];
    }

    std::string type = parse_required<std::string>(node, "type", "Texture type missing or invalid (line " + std::to_string(node.Mark().line) + ")");

    // Parse inline solid color texture
    if (type == "solid_color") {
        vec3 color = parse_required_vec3(node, "color", "Solid color 'color' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        textures.push_back(new solid_color(color));
        return textures.size() - 1;
    }

    // Parse inline checker texture
    if (type == "checker") {
        const YAML::Node& odd_node = node["odd"];
        const YAML::Node& even_node = node["even"];
        if (!odd_node || !even_node) {
            error_with_message("Checker texture must have 'odd' and 'even' fields (line " + std::to_string(node.Mark().line) + ")");
        }

        int odd_idx = parse_texture(odd_node, textures, texture_map);
        int even_idx = parse_texture(even_node, textures, texture_map);
        float scale = parse_optional<float>(node, "scale", 1.0f);

        textures.push_back(new checker(odd_idx, even_idx, scale));
        return textures.size() - 1;
    }

    // Parse inline image texture
    if (type == "image") {
        std::string filename = parse_required<std::string>(node, "filename", "Image texture 'filename' string missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        float hdr_gamma = parse_optional<float>(node, "gamma", 2.2f);
        float hdr_scale = parse_optional<float>(node, "scale", 1.0f);
        int desired_channels = parse_optional<int>(node, "channels", 0);
        bool flip_y = parse_optional<bool>(node, "flip_y", false);
        textures.push_back(new image(filename, hdr_gamma, hdr_scale, desired_channels, flip_y));
        return textures.size() - 1;
    }

    // Unknown texture type
    error_with_message("Unknown texture type '" + type + "' (line " + std::to_string(node.Mark().line) + ")");
    return -1; // Unreachable
}

/**
 * @brief Parse the texture list from the root YAML node
 * 
 * @param[in] config Root YAML node
 * @param[out] textures Textures in the scene
 * @param[out] texture_map Map of named textures
 */
void parse_texture_list(const YAML::Node& config, std::vector<_texture*>& textures, std::unordered_map<std::string, int>& texture_map) {
    // Check if textures are present
    const YAML::Node& tex_node = config["textures"];
    if (!tex_node) {
        return;
    }
    if (!tex_node.IsSequence()) {
        error_with_message("Textures must be a sequence (line " + std::to_string(tex_node.Mark().line) + ")");
    }

    // Parse each texture
    for (const YAML::Node& node : tex_node) {
        std::string name = parse_required<std::string>(node, "name", "Texture 'name' string missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        int tex_idx = parse_texture(node, textures, texture_map);
        texture_map[name] = tex_idx;
    }
}

/**
 * @brief Parse a material YAML node
 * 
 * @param[in] node Material YAML node
 * @param[in,out] textures Textures in the scene
 * @param[in,out] materials Materials in the scene
 * @param[in,out] texture_map Map of named textures
 * @param[in,out] material_map Map of named materials
 * @return Index of parsed material
 */
int parse_material(const YAML::Node& node, std::vector<_texture*>& textures, std::vector<material*>& materials, std::unordered_map<std::string, int>& texture_map, std::unordered_map<std::string, int>& material_map) {
    // Parse scalar material
    if (node.IsScalar()) {
        std::string name = node.as<std::string>();
        if (!material_map.count(name)) {
            error_with_message("Material '" + name + "' not found (line " + std::to_string(node.Mark().line) + ")");
        }
        return material_map[name];
    }

    std::string type = parse_required<std::string>(node, "type", "Material type missing or invalid (line " + std::to_string(node.Mark().line) + ")");

    // Parse inline lambertian material
    if (type == "lambertian") {
        const YAML::Node& tex_node = node["texture"];
        if (!tex_node) {
            error_with_message("Lambertian material must have 'texture' field (line " + std::to_string(node.Mark().line) + ")");
        }
        int tex_idx = parse_texture(tex_node, textures, texture_map);
        materials.push_back(new lambertian(tex_idx));
        return materials.size() - 1;
    }

    // Parse inline metal material
    if (type == "metal") {
        vec3 color = parse_required_vec3(node, "color", "Metal material 'color' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        float fuzz = parse_optional<float>(node, "fuzz", 0.0f);
        materials.push_back(new metal(color, fuzz));
        return materials.size() - 1;
    }

    // Parse inline glass material
    if (type == "dielectric") {
        float ior = parse_optional<float>(node, "ior", 1.5f);
        materials.push_back(new dielectric(ior));
        return materials.size() - 1;
    }

    // Parse inline diffuse light material
    if (type == "diffuse_light") {
        const YAML::Node& tex_node = node["texture"];
        if (!tex_node) {
            error_with_message("Diffuse light material must have 'texture' field (line " + std::to_string(node.Mark().line) + ")");
        }
        int tex_idx = parse_texture(tex_node, textures, texture_map);
        materials.push_back(new diffuse_light(tex_idx));
        return materials.size() - 1;
    }

    // Unknown material type
    error_with_message("Unknown material type '" + type + "' (line " + std::to_string(node.Mark().line) + ")");
    return -1; // Unreachable
}

/**
 * @brief Parse the material list from the root YAML node
 * 
 * @param[in] config Root YAML node
 * @param[out] textures Textures in the scene
 * @param[out] materials Materials in the scene
 * @param[out] texture_map Map of named textures
 * @param[out] material_map Map of named materials
 */
void parse_material_list(const YAML::Node& config, std::vector<_texture*>& textures, std::vector<material*>& materials, std::unordered_map<std::string, int>& texture_map, std::unordered_map<std::string, int>& material_map) {
    // Check if materials are present
    const YAML::Node& mat_node = config["materials"];
    if (!mat_node) {
        return;
    }
    if (!mat_node.IsSequence()) {
        error_with_message("Materials must be a sequence (line " + std::to_string(mat_node.Mark().line) + ")");
    }

    // Parse each material
    for (const YAML::Node& node : mat_node) {
        std::string name = parse_required<std::string>(node, "name", "Material 'name' string missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        int mat_idx = parse_material(node, textures, materials, texture_map, material_map);
        material_map[name] = mat_idx;
    }
}

/**
 * @brief Parse a hittable object YAML node
 * 
 * @param[in] node Object YAML node
 * @param[in,out] textures Textures in the scene
 * @param[in,out] materials Materials in the scene
 * @param[in,out] objects Objects in the scene
 * @param[in,out] texture_map Map of named textures
 * @param[in,out] material_map Map of named materials
 * @return Indices of parsed objects
 */
std::vector<int> parse_object(const YAML::Node& node, std::vector<_texture*>& textures, std::vector<material*>& materials, std::vector<hittable*>& objects, std::unordered_map<std::string, int>& texture_map, std::unordered_map<std::string, int>& material_map) {
    std::string type = parse_required<std::string>(node, "type", "Object type missing or invalid (line " + std::to_string(node.Mark().line) + ")");

    // Check if object has normal map
    int normal_idx = -1;
    const YAML::Node& norm_node = node["normal"];
    if (norm_node) {
        normal_idx = parse_texture(norm_node, textures, texture_map);
    }

    // Parse inline sphere object
    if (type == "sphere") {
        vec3 center = parse_required_vec3(node, "center", "Sphere 'center' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        vec3 center2 = parse_optional_vec3(node, "center2", center);
        float radius = parse_required<float>(node, "radius", "Sphere 'radius' float missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        
        const YAML::Node& mat_node = node["material"];
        if (!mat_node) {
            error_with_message("Sphere object must have 'material' field (line " + std::to_string(node.Mark().line) + ")");
        }
        int mat_idx = parse_material(mat_node, textures, materials, texture_map, material_map);

        objects.push_back(new sphere(center, center2, radius, mat_idx));
        objects[objects.size() - 1]->normal_idx = normal_idx;
        return {(int)(objects.size() - 1)};
    }

    // Parse inline UV sphere object
    if (type == "uv_sphere") {
        vec3 center = parse_required_vec3(node, "center", "UV sphere 'center' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        float radius = parse_required<float>(node, "radius", "UV sphere 'radius' float missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        float rings = parse_optional<int>(node, "rings", 5);
        float segments = parse_optional<int>(node, "segments", 10);
        float shade_smooth = parse_optional<bool>(node, "shade_smooth", false);
        
        const YAML::Node& mat_node = node["material"];
        if (!mat_node) {
            error_with_message("UV sphere object must have 'material' field (line " + std::to_string(node.Mark().line) + ")");
        }
        int mat_idx = parse_material(mat_node, textures, materials, texture_map, material_map);

        std::vector<tri*> sphere_tris = uv_sphere(center, radius, mat_idx, rings, segments, shade_smooth, normal_idx);
        std::vector<int> indices;
        for (int i = 0; i < sphere_tris.size(); i++) {
            sphere_tris[i]->normal_idx = normal_idx;
            objects.push_back(sphere_tris[i]);
            indices.push_back(objects.size() - 1);
        }
        return indices;
    }

    // Parse inline tri object
    if (type == "tri") {
        vec3 v0 = parse_required_vec3(node, "v0", "Tri 'v0' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        vec3 v1 = parse_required_vec3(node, "v1", "Tri 'v1' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        vec3 v2 = parse_required_vec3(node, "v2", "Tri 'v2' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        vec2 uv0 = parse_optional_vec2(node, "uv0", vec2(0, 1));
        vec2 uv1 = parse_optional_vec2(node, "uv1", vec2(1, 0));
        vec2 uv2 = parse_optional_vec2(node, "uv2", vec2(0, 0));
        vec3 normal = vec3::unit_vector(vec3::cross(v1 - v0, v2 - v0));

        const YAML::Node& mat_node = node["material"];
        if (!mat_node) {
            error_with_message("Tri object must have 'material' field (line " + std::to_string(node.Mark().line) + ")");
        }
        int mat_idx = parse_material(mat_node, textures, materials, texture_map, material_map);

        objects.push_back(new tri(v0, v1, v2, normal, normal, normal, uv0, uv1, uv2, mat_idx));
        objects[objects.size() - 1]->normal_idx = normal_idx;
        return {(int)(objects.size() - 1)};
    }

    // Parse inline quad
    if (type == "quad") {
        vec3 corner = parse_required_vec3(node, "corner", "Quad 'corner' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        vec3 u = parse_required_vec3(node, "edge1", "Quad 'edge1' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        vec3 v = parse_required_vec3(node, "edge2", "Quad 'edge2' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");

        const YAML::Node& mat_node = node["material"];
        if (!mat_node) {
            error_with_message("Quad object must have 'material' field (line " + std::to_string(node.Mark().line) + ")");
        }
        int mat_idx = parse_material(mat_node, textures, materials, texture_map, material_map);

        std::vector<tri*> quad_tris = quad(corner, u, v, mat_idx);
        std::vector<int> indices;
        for (int i = 0; i < quad_tris.size(); i++) {
            quad_tris[i]->normal_idx = normal_idx;
            objects.push_back(quad_tris[i]);
            indices.push_back(objects.size() - 1);
        }
        return indices;
    }

    // Parse inline box object
    if (type == "box") {
        vec3 corner1 = parse_required_vec3(node, "corner1", "Box 'corner1' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        vec3 corner2 = parse_required_vec3(node, "corner2", "Box 'corner2' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");

        const YAML::Node& mat_node = node["material"];
        if (!mat_node) {
            error_with_message("Box object must have 'material' field (line " + std::to_string(node.Mark().line) + ")");
        }
        int mat_idx = parse_material(mat_node, textures, materials, texture_map, material_map);

        std::vector<tri*> box_tris = box(corner1, corner2, mat_idx);
        std::vector<int> indices;
        for (int i = 0; i < box_tris.size(); i++) {
            box_tris[i]->normal_idx = normal_idx;
            objects.push_back(box_tris[i]);
            indices.push_back(objects.size() - 1);
        }
        return indices;
    }

    // Parse inline model object
    if (type == "model") {
        std::string filename = parse_required<std::string>(node, "filename", "Model object 'filename' string missing or invalid (line " + std::to_string(node.Mark().line) + ")");
        float scale = parse_optional<float>(node, "scale", 1.0f);

        const YAML::Node& mat_node = node["material"];
        if (!mat_node) {
            error_with_message("Model object must have 'material' field (line " + std::to_string(node.Mark().line) + ")");
        }
        int mat_idx = parse_material(mat_node, textures, materials, texture_map, material_map);

        std::vector<tri*> model_tris = load_model(filename, mat_idx, scale);
        std::vector<int> indices;
        for (int i = 0; i < model_tris.size(); i++) {
            model_tris[i]->normal_idx = normal_idx;
            objects.push_back(model_tris[i]);
            indices.push_back(objects.size() - 1);
        }
        return indices;
    }

    // Parse inline translated object
    if (type == "translate") {
        vec3 offset = parse_required_vec3(node, "offset", "Translate 'offset' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");

        const YAML::Node& obj_node = node["object"];
        if (!obj_node) {
            error_with_message("Translate object must have 'object' field (line " + std::to_string(node.Mark().line) + ")");
        }
        std::vector<int> obj_idx = parse_object(obj_node, textures, materials, objects, texture_map, material_map);
        for (int idx : obj_idx) {
            objects[idx]->translate(offset);
        }
        return obj_idx;
    }

    // Parse inline rotated object
    if (type == "rotate") {
        vec3 rotations = parse_required_vec3(node, "angles", "Rotate 'angles' vector missing or invalid (line " + std::to_string(node.Mark().line) + ")");

        const YAML::Node& obj_node = node["object"];
        if (!obj_node) {
            error_with_message("Rotate object must have 'object' field (line " + std::to_string(node.Mark().line) + ")");
        }
        std::vector<int> obj_idx = parse_object(obj_node, textures, materials, objects, texture_map, material_map);

        vec3 anchor = calculate_center(objects, obj_idx);
        const YAML::Node& anchor_node = node["anchor"];
        if (anchor_node && anchor_node.IsSequence()) {
            anchor = parse_required_vec3(node, "anchor", "Rotate 'anchor' vector invalid (line " + std::to_string(node.Mark().line) + ")");
        }

        for (int idx : obj_idx) {
            objects[idx]->rotate(rotations, anchor);
        }
        return obj_idx;
    }

    // Unknown object type
    error_with_message("Unknown object type '" + type + "' (line " + std::to_string(node.Mark().line) + ")");
    return {}; // Unreachable
}

/**
 * @brief Parse the object list from the root YAML node
 * 
 * @param[in] config Root YAML node
 * @param[out] textures Textures in the scene
 * @param[out] materials Materials in the scene
 * @param[out] objects Objects in the scene
 * @param[out] texture_map Map of named textures
 * @param[out] material_map Map of named materials
 */
void parse_object_list(const YAML::Node& config, std::vector<_texture*>& textures, std::vector<material*>& materials, std::vector<hittable*>& objects, std::unordered_map<std::string, int>& texture_map, std::unordered_map<std::string, int>& material_map) {
    // Check if objects are present
    const YAML::Node& obj_node = config["objects"];
    if (!obj_node || !obj_node.IsSequence()) {
        error_with_message("Objects missing or invalid");
    }

    // Parse each object
    for (const YAML::Node& node : obj_node) {
        std::vector<int> obj_idx = parse_object(node, textures, materials, objects, texture_map, material_map);
    }
}

/**
 * @brief Parse the environment texture from the root YAML node
 * 
 * @param[in] config Root YAML node
 * @param[in,out] textures Textures in the scene
 * @param[in, out] texture_map Named textures in the scene
 * @return Index of parsed environment texture
 */
int parse_environment(const YAML::Node& config, std::vector<_texture*>& textures, std::unordered_map<std::string, int>& texture_map) {
    // Check if environment is present
    const YAML::Node& env_node = config["environment"];
    if (!env_node || !env_node.IsMap()) {
        error_with_message("Environment missing or invalid");
    }

    // Parse environment texture
    const YAML::Node& tex_node = env_node["texture"];
    if (!tex_node) {
        error_with_message("Environment must have 'texture' field");
    }
    int env_idx = parse_texture(tex_node, textures, texture_map);
    return env_idx;
}

/**
 * @brief Load a scene from a YAML file
 * 
 * @param scene_file YAML scene file
 * @return Parsed scene
 */
std::unique_ptr<scene> load_scene(const std::string& scene_file) {
    std::cout << "Loading scene file '" << scene_file << "'...\n";

    // Open scene file
    YAML::Node config;
    try {
        config = YAML::LoadFile(scene_file);
    } catch (const std::exception& e) {
        error_with_message("Failed to load scene file: " + std::string(e.what()));
    }

    // Parse camera and render settings
    camera* cam = parse_settings(config);
    std::cout << " - Parsed camera and render settings.\n";

    // Create vectors to store objects, materials, and textures
    std::vector<hittable*> objects;
    std::vector<material*> materials;
    std::vector<_texture*> textures;

    // Parse textures
    std::unordered_map<std::string, int> texture_map;
    parse_texture_list(config, textures, texture_map);
    std::cout << " - Parsed " << textures.size() << " textures.\n";

    // Parse materials
    std::unordered_map<std::string, int> material_map;
    parse_material_list(config, textures, materials, texture_map, material_map);
    std::cout << " - Parsed " << materials.size() << " materials.\n";

    // Parse objects
    parse_object_list(config, textures, materials, objects, texture_map, material_map);
    std::cout << " - Parsed " << objects.size() << " objects.\n";

    // Parse environment
    int background_idx = parse_environment(config, textures, texture_map);
    std::cout << " - Parsed environment texture.\n";

    // Create scene
    return std::make_unique<scene>(objects, materials, textures, cam, background_idx);
}

#endif