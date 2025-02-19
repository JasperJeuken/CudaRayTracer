/**
 * @file bvh.cuh
 * @author Jasper Jeuken
 * @brief Defines a bounding volume hierarchy (BVH) for ray tracing
 */
#ifndef BVH_H
#define BVH_H

#include "aabb.cuh"
#include "hittable.cuh"

/**
 * @struct bvh4_tree
 * @brief Flattened BVH4 tree for ray tracing
 * 
 */
struct bvh4_tree {
    float3* min_bounds; ///< Minimum bounds of each node
    float3* max_bounds; ///< Maximum bounds of each node
    int* children; ///< Children of each node
    int* object_index; ///< Index of object in leaf node
    int node_count; ///< Number of nodes in the tree

    /**
     * @brief Construct a new BVH4 tree object with a given number of nodes
     * 
     * @param[in] count Number of nodes in the tree
     */
    bvh4_tree(int count) : node_count(count) {
        min_bounds = new float3[count];
        max_bounds = new float3[count];
        children = new int[count];
        object_index = new int[count];
    }
};

/**
 * @struct bvh4_node_recursive
 * @brief Recursive BVH4 node for construction
 * 
 */
struct bvh4_node_recursive : public hittable {
public:
    hittable* children[4]; ///< Children of the node
    aabb box; ///< Bounding box of the node
    bool is_leaf[4]; ///< Flags indicating if children are leaf nodes

    /**
     * @brief Default constructor
     * 
     * @return Empty node
     */
    __host__ bvh4_node_recursive() : box(), hittable(object_type::BVH, -1) {
        for (int i = 0; i < 4; i++) {
            children[i] = nullptr;
            is_leaf[i] = false;
        }
    }

    /**
     * @brief Construct a new BVH4 node from a list of objects
     * 
     * @param[in] objects List of objects to build the BVH from
     * @param[in] start Start index of the object list
     * @param[in] end End index of the object list
     * @return BVH4 node
     */
    __host__ bvh4_node_recursive(std::vector<hittable*> objects, size_t start, size_t end) : hittable(object_type::BVH, -1) {

        // Filter visible objects
        std::vector<hittable*> visible_objects;
        for (size_t i = start; i < end; i++) {
            if (objects[i]->visible) {
                visible_objects.push_back(objects[i]);
            }
        }
        size_t object_span = visible_objects.size();

        // Empty node: set all children to nullptr and bounding box to empty
        if (object_span == 0) {
            for (int i = 0; i < 4; i++) {
                children[i] = nullptr;
                is_leaf[i] = false;
            }
            box = aabb::empty;
            return;
        }

        // Leaf node: add all objects as children and calculate bounding box
        if (object_span <= 4) {
            for (size_t i = 0; i < object_span; i++) {
                children[i] = visible_objects[i];
                is_leaf[i] = true;
            }
            for (size_t i = object_span; i < 4; i++) {
                children[i] = nullptr;
                is_leaf[i] = false;
            }
            box = compute_bounding_box(visible_objects, 0, object_span);
            return;
        }

        // Internal node: all children are subtrees
        // Calculate bounding box of all objects
        aabb bbox = compute_bounding_box(visible_objects, 0, object_span);
        int axis = bbox.longest_axis();

        // Sort objects along longest axis
        auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;
        std::sort(visible_objects.begin(), visible_objects.end(), comparator);

        // Split objects into 4 groups
        size_t mid1 = 1 * object_span / 4;
        size_t mid2 = 2 * object_span / 4;
        size_t mid3 = 3 * object_span / 4;

        // Recursively build child nodes
        children[0] = new bvh4_node_recursive(visible_objects, 0, mid1);
        children[1] = new bvh4_node_recursive(visible_objects, mid1, mid2);
        children[2] = new bvh4_node_recursive(visible_objects, mid2, mid3);
        children[3] = new bvh4_node_recursive(visible_objects, mid3, object_span);

        // Set leaf flags
        for (int i = 0; i < 4; i++) {
            is_leaf[i] = false;
        }

        // Update node bounding box
        box = aabb::empty;
        for (int i = 0; i < 4; i++) {
            box = aabb(box, static_cast<bvh4_node_recursive*>(children[i])->box);
        }
    }

private:

    /**
     * @brief Compare two hittables along a given axis
     * 
     * @param[in] a First hittable
     * @param[in] b Second hittable
     * @param[in] axis Axis to compare along
     * @return Whether a is smaller than b along the axis
     */
    __host__ static bool box_compare(const hittable* a, const hittable* b, int axis) {
        interval a_axis_interval = a->bounding_box()[axis];
        interval b_axis_interval = b->bounding_box()[axis];
        return a_axis_interval.min < b_axis_interval.min;
    }

    /**
     * @brief Compare two hittables along the x-axis
     * 
     * @param[in] a First hittable
     * @param[in] b Second hittable
     * @return Whether a is smaller than b along the x-axis
     */
    __host__ static bool box_x_compare(const hittable* a, const hittable* b) {
        return box_compare(a, b, 0);
    }

    /**
     * @brief Compare two hittables along the y-axis
     * 
     * @param[in] a First hittable
     * @param[in] b Second hittable
     * @return Whether a is smaller than b along the y-axis
     */
    __host__ static bool box_y_compare(const hittable* a, const hittable* b) {
        return box_compare(a, b, 1);
    }

    /**
     * @brief Compare two hittables along the z-axis
     * 
     * @param[in] a First hittable
     * @param[in] b Second hittable
     * @return Whether a is smaller than b along the z-axis
     */
    __host__ static bool box_z_compare(const hittable* a, const hittable* b) {
        return box_compare(a, b, 2);
    }

    /**
     * @brief Compute the bounding box of a list of objects
     * 
     * @param[in] objects List of objects
     * @param[in] start Start index
     * @param[in] end End index
     * @return Bounding box of the objects
     */
    aabb compute_bounding_box(const std::vector<hittable*>& objects, size_t start, size_t end) {
        aabb bbox = aabb::empty;
        for (size_t i = start; i < end; i++) {
            bbox = aabb(bbox, objects[i]->bounding_box());
        }
        return bbox;
    }
};

/**
 * @brief Flatten a recursive BVH4 tree into a flat tree
 * 
 * @param[in] node Root node of the recursive tree
 * @param[out] tree Output flat tree
 * @param[in] current_index Current index in the flat tree
 * @param[in] objects List of objects in the scene
 * @return Index of the current node in the flat tree
 */
int flatten_tree(const bvh4_node_recursive* node, bvh4_tree& tree, int& current_index, const std::vector<hittable*>& objects) {
    int node_index = current_index++;
    for (int i = 0; i < 4; i++) {
        int child_index = node_index * 4 + i;
        if (node->children[i]) {
            if (node->is_leaf[i]) {
                // Populate leaf node
                auto it = std::find(objects.begin(), objects.end(), node->children[i]);
                if (it == objects.end()) throw std::runtime_error("Object in BVH4 tree not found");
                int obj_index = std::distance(objects.begin(), it);

                aabb box = node->children[i]->bounding_box();
                tree.min_bounds[child_index] = box.min().as_float3();
                tree.max_bounds[child_index] = box.max().as_float3();
                tree.object_index[child_index] = obj_index;
                tree.children[child_index] = -1;
            } else {
                // Populate internal node
                const bvh4_node_recursive* child = static_cast<const bvh4_node_recursive*>(node->children[i]);
                tree.min_bounds[child_index] = child->box.min().as_float3();
                tree.max_bounds[child_index] = child->box.max().as_float3();
                tree.object_index[child_index] = -1;
                tree.children[child_index] = flatten_tree(child, tree, current_index, objects);
            }
        } else {
            // Populate empty node
            tree.min_bounds[child_index] = make_float3(h_infinity, h_infinity, h_infinity);
            tree.max_bounds[child_index] = make_float3(-h_infinity, -h_infinity, -h_infinity);
            tree.object_index[child_index] = -1;
            tree.children[child_index] = -1;
        }
    }
    return node_index;
}

/**
 * @brief Count the number of nodes in a recursive BVH4 tree
 * 
 * @param[in] node Root node of the recursive tree
 * @return Number of nodes in the tree
 */
int count_bvh4_nodes(const bvh4_node_recursive* node) {
    if (node == nullptr) return 0;

    size_t count = 1;

    for (int i = 0; i < 4; i++) {
        if (!node->is_leaf[i] && node->children[i] != nullptr) {
            count += count_bvh4_nodes(static_cast<bvh4_node_recursive*>(node->children[i]));
        }
    }
    return count;
}

/**
 * @brief Debugging function for printing a recursive BVH4 tree
 * 
 * @param node Node to print
 * @param depth Current depth
 * @param index Current node index
 */
void print_recursive_bvh4(const bvh4_node_recursive* node, int depth = 0, int index = 0) {
    if (!node) return;

    std::string indent(depth * 4, ' ');

    std::cout << indent << "Node " << index << " | " << node->box << "\n";

    for (int i = 0; i < 4; i++) {
        if (node->children[i]) {
            if (node->is_leaf[i]) {
                const hittable* obj = node->children[i];
                std::cout << indent << "  - Leaf[" << i << "] -> " << obj << " | " << obj->bbox << "\n";
            } else {
                std::cout << indent << "  - Child[" << i << "] -> Subtree\n";
                print_recursive_bvh4(static_cast<bvh4_node_recursive*>(node->children[i]), depth + 1, index * 4 + i);
            }
        } else {
            std::cout << indent << "  - Child[" << i << "] -> nullptr\n";
        }
    }
}

/**
 * @brief Create a BVH4 tree from a list of objects
 * 
 * @param[in] objects List of objects in the scene
 * @return BVH4 tree
 */
bvh4_tree create_bvh4(std::vector<hittable*>& objects) {
    std::cout << "Creating BVH4 tree...\n";
    // Create recursive tree
    bvh4_node_recursive* root = new bvh4_node_recursive(objects, 0, objects.size());
    int node_count = count_bvh4_nodes(root) * 4;

    // Create flat tree
    bvh4_tree tree(node_count);
    int current_index = 0;
    flatten_tree(root, tree, current_index, objects);
    return tree;
}

/**
 * @brief Allocate a BVH4 tree on the device
 * 
 * @param[in] tree Host BVH4 tree
 * @return Device BVH4 tree
 */
bvh4_tree* allocate_bvh4(const bvh4_tree& tree) {
    // Allocate arrays
    float3* d_min_bounds;
    float3* d_max_bounds;
    int* d_children;
    int* d_object_index;
    checkCudaErrors(cudaMalloc(&d_min_bounds, tree.node_count * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&d_max_bounds, tree.node_count * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&d_children, tree.node_count * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_object_index, tree.node_count * sizeof(int)));

    // Copy data
    checkCudaErrors(cudaMemcpy(d_min_bounds, tree.min_bounds, tree.node_count * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_max_bounds, tree.max_bounds, tree.node_count * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_children, tree.children, tree.node_count * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_object_index, tree.object_index, tree.node_count * sizeof(int), cudaMemcpyHostToDevice));

    // Create device BVH tree
    bvh4_tree* d_bvh;
    checkCudaErrors(cudaMalloc(&d_bvh, sizeof(bvh4_tree)));
    bvh4_tree temp_tree = tree;
    temp_tree.min_bounds = d_min_bounds;
    temp_tree.max_bounds = d_max_bounds;
    temp_tree.children = d_children;
    temp_tree.object_index = d_object_index;
    temp_tree.node_count = tree.node_count;
    checkCudaErrors(cudaMemcpy(d_bvh, &temp_tree, sizeof(bvh4_tree), cudaMemcpyHostToDevice));

    return d_bvh;
}

/**
 * @brief Free a BVH4 tree on the device
 * 
 * @param[in] d_tree Device BVH4 tree
 * @param[in] node_count Number of nodes in the tree
 */
void free_bvh4(bvh4_tree* d_tree, int node_count) {
    bvh4_tree* h_tree = new bvh4_tree(node_count);
    checkCudaErrors(cudaMemcpy(h_tree, d_tree, sizeof(bvh4_tree), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(h_tree->min_bounds));
    checkCudaErrors(cudaFree(h_tree->max_bounds));
    checkCudaErrors(cudaFree(h_tree->children));
    checkCudaErrors(cudaFree(h_tree->object_index));
    checkCudaErrors(cudaFree(d_tree));

    delete h_tree;
}

/**
 * @brief Traverse a BVH4 tree to find the closest hit
 * 
 * @param[in] r Ray to test
 * @param[in] ray_t Interval on the ray
 * @param[out] rec Hit record to populate
 * @param[in] rand_state Random state for random number generation
 * @param[in] tree BVH4 tree to traverse
 * @param[in] objects List of objects in the scene
 * @param[in] textures List of textures in the scene
 * @return Whether a hit was found
 */
__device__ bool bvh4_hit(const ray& r, interval ray_t, hit_record& rec, curandState* rand_state, const bvh4_tree* tree, hittable** objects, _texture** textures){
    int stack[256];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    bool hit_anything = false;
    float closest_t = ray_t.max;

    while (stack_ptr > 0) {
        int node_index = stack[--stack_ptr];
        if (node_index == -1) continue;

        for (int i = 0; i < 4; i++) {
            int child_index = node_index * 4 + i;
            
            // Check if bounding box is hit
            float3 child_min = tree->min_bounds[child_index];
            float3 child_max = tree->max_bounds[child_index];
            aabb child_box(child_min, child_max);
            if (!child_box.hit(r, ray_t)) continue;

            // Get node properties
            int object_index = tree->object_index[child_index];
            int children_index = tree->children[child_index];

            // If leaf node: check object hit
            if (object_index >= 0) {
                hittable* obj = objects[object_index];
                if (obj->hit(r, ray_t, rec, rand_state, textures)) {
                    hit_anything = true;
                    closest_t = rec.t;
                    ray_t.max = closest_t;
                }
            }

            // If internal node: add children to stack
            if (children_index >= 0) {
                stack[stack_ptr++] = children_index;
            }
        }
    }
    return hit_anything;
}


#endif