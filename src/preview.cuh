/**
 * @file preview.cuh
 * @author Jasper Jeuken
 * @brief Defines a preview window for the renderer
 */
#ifndef PREVIEW_H
#define PREVIEW_H

#define SDL_MAIN_HANDLED
#define GLT_IMPLEMENTATION

#include <SDL.h>
#include "GL/glew.h"
#include "gltext.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>

#include "scene.cuh"


GLuint texture;
cudaGraphicsResource* cudaResource;


/**
 * @brief Compile a shader
 * 
 * @param[in] shaderSource Shader source code
 * @param[in] shaderType Shader type
 * @return Compiled shader ID
 */
GLuint compile_shader(const char* shaderSource, GLenum shaderType) {
    GLuint shaderID = glCreateShader(shaderType);
    glShaderSource(shaderID, 1, &shaderSource, nullptr);
    glCompileShader(shaderID);

    // Check for compilation errors
    GLint success;
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shaderID, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Failed: " << infoLog << std::endl;
        return 0;
    }

    return shaderID;
}

/**
 * @brief Link a program
 * 
 * @param[in] vertexShaderID Shader ID of the vertex shader
 * @param[in] fragmentShaderID Shader ID of the fragment shader
 * @return Linked program ID
 */
GLuint link_program(GLuint vertexShaderID, GLuint fragmentShaderID) {
    GLuint programID = glCreateProgram();
    glAttachShader(programID, vertexShaderID);
    glAttachShader(programID, fragmentShaderID);
    glLinkProgram(programID);

    // Check for linking errors
    GLint success;
    glGetProgramiv(programID, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(programID, 512, nullptr, infoLog);
        std::cerr << "Program Linking Failed: " << infoLog << std::endl;
        return 0;
    }

    return programID;
}

// Shader which displays a texture flipped vertically
const char* vertexShaderSource = R"(
#version 330 core
const vec2 quad_vertices[4] = vec2[4](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
);
const vec2 tex_coords[4] = vec2[4](
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 0.0)
);
out vec2 TexCoords;
void main() {
    gl_Position = vec4(quad_vertices[gl_VertexID], 0.0, 1.0);
    TexCoords = tex_coords[gl_VertexID];
}
)";

// Shader which applies gamma correction
const char* fragmentShaderSource = R"(
#version 330 core
uniform sampler2D renderedTexture;
uniform float gamma;
in vec2 TexCoords;
out vec4 FragColor;
void main() {
    vec4 color = texture(renderedTexture, TexCoords);
    color = pow(color, vec4(1.0 / gamma)); // Gamma correction
    FragColor = color;
}
)";
    
/**
 * @brief Draw a texture to the screen
 * 
 * @param[in] textureID ID of the texture to draw
 * @param[in] available_width Available width
 * @param[in] available_height Available height
 * @param[in] texture_width Texture width
 * @param[in] texture_height Texture height
 */
void draw_texture(GLuint textureID, int available_width, int available_height, int texture_width, int texture_height) {
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Ensure the viewport is set correctly
    float aspect_ratio = float(texture_width) / texture_height;
    float window_aspect_ratio = float(available_width) / available_height;
    
    if (aspect_ratio > window_aspect_ratio) {
        float scale = float(available_width) / texture_width;
        glViewport(0, (available_height - texture_height * scale) / 2, available_width, texture_height * scale);
    } else {
        float scale = float(available_height) / texture_height;
        glViewport((available_width - texture_width * scale) / 2, 0, texture_width * scale, available_height);
    }
    
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

/**
 * @brief Initialize a texture
 * 
 * @param[in] width Image width
 * @param[in] height Image height
 */
void init_texture(int width, int height) {
    // Generate and bind a texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate texture memory
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

    // Register texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

/**
 * @brief Pad a vec3 buffer to a 4x float buffer with an alpha channel
 * 
 * @param data Original buffer (vec3)
 * @param padded_data Padded buffer (float)
 * @param width Image width
 * @param height Image height
 */
__global__ void copy_buffer_to_padded_buffer(vec3* data, float* padded_data, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int index = j * width + i;
    int padded_index = index * 4;
    padded_data[padded_index + 0] = data[index][0];
    padded_data[padded_index + 1] = data[index][1];
    padded_data[padded_index + 2] = data[index][2];
    padded_data[padded_index + 3] = 1.0f;
}

/**
 * @brief Copy CUDA buffer to OpenGL texture
 * 
 * @param data CUDA buffer
 * @param width Image width
 * @param height Image height
 * @param tx Number of threads in x direction
 * @param ty Number of threads in y direction
 */
void copy_buffer_to_texture(vec3* data, int width, int height, int tx = 8, int ty = 8) {
    cudaArray* textureArray;

    // Map OpenGL to CUDA
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaResource, 0, 0));

    // Copy vec3 buffer to a float buffer with padded alpha channel
    float* padded_data;
    checkCudaErrors(cudaMalloc(&padded_data, width * height * sizeof(float) * 4));
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    copy_buffer_to_padded_buffer<<<blocks, threads>>>(data, padded_data, width, height);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // Copy data to texture
    checkCudaErrors(cudaMemcpy2DToArray(textureArray, 0, 0, padded_data, width * sizeof(float) * 4, width * sizeof(float) * 4, height, cudaMemcpyDeviceToDevice));

    // Unmap resource
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource, 0));

    // Free memory
    checkCudaErrors(cudaFree(padded_data));
}


/**
 * @brief Render a scene with a preview window
 * 
 * @param[in] sc Scene to render
 * @param[in] buf Buffer to render to
 * @param[in] tx Number of threads in x direction
 * @param[in] ty Number of threads in y direction
 * @return Render duration (seconds)
 */
float render_with_preview(std::unique_ptr<scene>& sc, std::unique_ptr<buffer>& buf, int tx = 8, int ty = 8) {
    // Calculate window size
    int max_size = 1000;
    float aspect_ratio = float(buf->width) / buf->height;
    int window_width, window_height;
    if (aspect_ratio > 1.0f) {
        window_width = max_size;
        window_height = int(max_size / aspect_ratio);
    } else  {
        window_width = int(max_size * aspect_ratio);
        window_height = max_size;
    }

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return -1;

    // Create window
    SDL_Window* window = SDL_CreateWindow("Preview", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!window) return -1;

    // Create OpenGL context
    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    if (!glContext) return -1;
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    if (GLenum err = glGetError() != GL_NO_ERROR) {
        std::cerr << "OpenGL error" << err << std::endl;
        return -1;
    };
    
    // Initialize GLEW
    if (glewInit() != GLEW_OK) return -1;

    // Initialize GL text
    if (!gltInit()) return -1;

    // Create texts
    GLTtext* progress_text = gltCreateText();
    gltSetText(progress_text, "Progress: 0.0%");

    // Compile shader
    GLuint vertex_shader = compile_shader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragment_shader = compile_shader(fragmentShaderSource, GL_FRAGMENT_SHADER);
    GLuint shader_program = link_program(vertex_shader, fragment_shader);

    // Set shader uniforms
    glUseProgram(shader_program);
    glUniform1f(glGetUniformLocation(shader_program, "gamma"), buf->gamma);

    // Clean up shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    // Set OpenGL state
    glViewport(0, 0, window_width, window_height);
    glClearColor(0, 0, 0, 1);

    // Create texture
    init_texture(buf->width, buf->height);

    // Start render
    const int samples_per_iteration = 1;
    int samples_done = 0;
    float render_duration = 0;
    std::atomic<bool> window_running(true);
    std::atomic<int> sample_ready(-1);
    std::thread render_thread([&]() {
        while (samples_done < buf->samples_per_pixel) {
            // Wait until texture is updated
            if (sample_ready.load() >= 0 && window_running.load()) continue;

            // Render samples
            int samples = minf(samples_per_iteration, buf->samples_per_pixel - samples_done);
            render_duration += render(sc, buf, tx, ty, samples, samples_done);
            samples_done += samples;

            // Set flag to update texture
            sample_ready.store(samples_done);

            // Sleep for a while
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Run window
    SDL_Event event;
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    while (window_running.load()) {

        // Check events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) window_running.store(false);
        }

        int curr_sample = sample_ready.load();
        if (curr_sample >= 0) {
            glClear(GL_COLOR_BUFFER_BIT);
            glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            
            // Copy curent buffer to texture
            copy_buffer_to_texture(buf->color, buf->width, buf->height);
            sample_ready.store(-1);
            
            // Draw texture
            glUseProgram(shader_program);
            draw_texture(texture, window_width, window_height, buf->width, buf->height);

            // Update progress text
            std::stringstream ss;
            ss << "Sample " << curr_sample << "/" << buf->samples_per_pixel << " (" << std::fixed << std::setprecision(1) << (float)curr_sample / buf->samples_per_pixel * 100 << "%)";
            std::string progress = ss.str();

            // Draw text
            gltBeginDraw();
            gltColor(1.0f, 1.0f, 1.0f, 1.0f);
            gltSetText(progress_text, progress.c_str());
            gltDrawText2D(progress_text, 10.0f, 5.0f, 1.5f);
            gltEndDraw();

            glFlush();
            glFinish();

            SDL_GL_SwapWindow(window);
        }
    }

    // Close window
    checkCudaErrors(cudaGraphicsUnregisterResource(cudaResource));
    glDeleteTextures(1, &texture);
    glDeleteProgram(shader_program);
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << " - Closed preview window.\n";

    // Wait for render to finish
    render_thread.join();

    return render_duration;
}

#endif