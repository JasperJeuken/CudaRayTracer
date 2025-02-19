/**
 * @file main.cu
 * @author Jasper Jeuken
 * @brief Defines the main entry point of the program
 */
#include "command_parse.cuh"
#include "scene.cuh"
#include "render.cuh"
#include "preview.cuh"
#include "denoise.cuh"


/**
 * @brief Main entry point of the program
 * 
 * @param[in] argc Argument count
 * @param[in] argv Command line arguments
 * @return Exit code
 */
int main(int argc, char** argv) {
    try {
        // Parse command line arguments
        parsed_args args = parse_command_line_args(argc, argv);

        // Load scene
        std::unique_ptr<scene> scene = load_scene(args.scene_file);

        // Render the image
        std::cout << "Starting render...\n";
        float render_duration = 0.0f;
        std::unique_ptr<buffer> buffer = create_buffer(scene);
        if (args.preview) {
            render_duration = render_with_preview(scene, buffer);
        } else {
            render_duration = render(scene, buffer);
        }
        if (render_duration < 0) {
            error_with_message("Render failed");
        }
        std::cout << " - Render completed: " << render_duration << " seconds" << std::endl;

        // buffer->copy_to_host();
        // std::cerr << buffer->color[0] << std::endl;

        // Denoise
        denoise(buffer);

        // Save the desired render passes
        buffer->save_passes(args.output_path, args.output_format, args.render_passes);

    } catch (const std::exception& e) {
        error_with_message(e.what());
    }

    std::cout << "Done.\n";
    return 0;
}
