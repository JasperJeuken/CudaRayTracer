/**
 * @file objects.cuh
 * @author Jasper Jeuken
 * @brief Defines functions to create objects
 */
#ifndef OBJECTS_H
#define OBJECTS_H

#include "hittable.cuh"


/**
 * @brief Create a UV sphere using triangles
 * 
 * A UV sphere is a sphere that is tessellated into triangles. 
 * The sphere is centered at the given center and has the given radius. 
 * The sphere is tessellated into rings and segments.
 * The sphere can be shaded smooth or flat. 
 * The sphere can be visible or invisible.
 * 
 * @param[in] center Sphere center
 * @param[in] radius Sphere radius
 * @param[in] mat_idx Material index
 * @param[in] rings Number of rings
 * @param[in] segments Number of segments
 * @param[in] shade_smooth Whether to shade smooth
 * @param[in] visible Visibility
 * @return Vector of triangles forming the sphere 
 */
__host__ std::vector<tri*> uv_sphere(vec3 center, float radius, int mat_idx, int rings = 5, int segments = 10, bool shade_smooth = false, bool visible = true) {
    // Create vectors
    std::vector<vec3> verts;
    std::vector<vec3> normals;
    std::vector<vec2> uvs;
    std::vector<tri*> tris;

    // Generate vertices, normals, and UVs
    for (int lat = 0; lat <= rings; lat++) {
        float theta = h_pi * lat / rings;
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);

        for (int lon = 0; lon <= segments; lon++) {
            float phi = 2.0f * h_pi * lon / segments;
            float sin_phi = sinf(phi);
            float cos_phi = cosf(phi);

            // Vertex position
            vec3 vert = center + radius * vec3(
                sin_theta * cos_phi,
                cos_theta,
                sin_theta * sin_phi
            );
            verts.push_back(vert);

            // Vertex normal
            vec3 normal = vec3::unit_vector(vert - center);
            normals.push_back(normal);

            // UV coordinates
            vec2 uv(
                phi / (2.0f * h_pi),
                (1.0f - cos_theta) / 2.0f
            );
            uvs.push_back(uv);
        }
    }

    // Generate triangles
    for (int lat = 0; lat < rings; lat++) {
        for (int lon = 0; lon < segments; lon++) {
            int v0 = lat * (segments + 1) + lon;
            int v1 = v0 + 1;
            int v2 = v0 + (segments + 1);
            int v3 = v2 + 1;

            tris.push_back(new tri(verts[v0], verts[v2], verts[v1], normals[v0], normals[v2], normals[v1], uvs[v0], uvs[v2], uvs[v1], mat_idx, shade_smooth, visible));
            tris.push_back(new tri(verts[v1], verts[v2], verts[v3], normals[v1], normals[v2], normals[v3], uvs[v1], uvs[v2], uvs[v3], mat_idx, shade_smooth, visible));
        }
    }
    return tris;
}

/**
 * @brief Create a quad using triangles
 * 
 * @param corner Corner point
 * @param u First edge
 * @param v Second edge
 * @param mat_idx Material index
 * @param visible Visibility
 * @return Vector of triangles forming the quad
 */
__host__ std::vector<tri*> quad(point3 corner, vec3 u, vec3 v, int mat_idx, bool visible = true) {
    std::vector<tri*> tris;
    vec3 normal = vec3::unit_vector(vec3::cross(u, v));
    tris.push_back(new tri(corner, corner + u, corner + u + v, normal, normal, normal, vec2(0, 0), vec2(1, 0), vec2(1, 1), mat_idx, false, visible));
    tris.push_back(new tri(corner, corner + u + v, corner + v, normal, normal, normal, vec2(0, 0), vec2(1, 1), vec2(0, 1), mat_idx, false, visible));
    return tris;
}

/**
 * @brief Create a box using triangles
 * 
 * @param corner1 First corner
 * @param corner2 Second corner
 * @param mat_idx Material index
 * @param visible Visibility
 * @return Vector of triangles forming the box
 */
__host__ std::vector<tri*> box(point3 corner1, point3 corner2, int mat_idx, bool visible = true) {
    std::vector<tri*> tris;

    // Handle any order of input points
    point3 min = point3(fmin(corner1.x(), corner2.x()), fmin(corner1.y(), corner2.y()), fmin(corner1.z(), corner2.z()));
    point3 max = point3(fmax(corner1.x(), corner2.x()), fmax(corner1.y(), corner2.y()), fmax(corner1.z(), corner2.z()));

    // Define directions
    vec3 dx = vec3(max.x() - min.x(), 0, 0);
    vec3 dy = vec3(0, max.y() - min.y(), 0);
    vec3 dz = vec3(0, 0, max.z() - min.z());

    // Add quads
    std::vector<tri*> tris1 = quad(min, dx, dy, mat_idx, visible);
    std::vector<tri*> tris2 = quad(min, dx, dz, mat_idx, visible);
    std::vector<tri*> tris3 = quad(min, dy, dz, mat_idx, visible);
    std::vector<tri*> tris4 = quad(max, -dx, -dy, mat_idx, visible);
    std::vector<tri*> tris5 = quad(max, -dx, -dz, mat_idx, visible);
    std::vector<tri*> tris6 = quad(max, -dy, -dz, mat_idx, visible);

    // Append triangles
    tris.insert(tris.end(), tris1.begin(), tris1.end());
    tris.insert(tris.end(), tris2.begin(), tris2.end());
    tris.insert(tris.end(), tris3.begin(), tris3.end());
    tris.insert(tris.end(), tris4.begin(), tris4.end());
    tris.insert(tris.end(), tris5.begin(), tris5.end());
    tris.insert(tris.end(), tris6.begin(), tris6.end());

    return tris;
}

/**
 * @brief Calculate the center point of a list of objects
 * 
 * @param objects Objects in scene
 * @param indices Indices of objects to calculate center of
 * @return Calculated center point 
 */
point3 calculate_center(std::vector<hittable*> objects, std::vector<int> indices) {
    // Skip if no objects are selected
    if (indices.empty()) {
        return vec3(0, 0, 0);
    }

    // Create combined bounding box
    aabb combined_bbox = objects[indices[0]]->bounding_box();
    for (int i = 1; i < indices.size(); i++) {
        combined_bbox = aabb(combined_bbox, objects[indices[i]]->bounding_box());
    }
    return combined_bbox.center();
}

#endif