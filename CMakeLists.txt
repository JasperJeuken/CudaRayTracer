cmake_minimum_required(VERSION 3.20)
project(CudaTracer LANGUAGES CXX CUDA)

# Enable FetchContent
include(FetchContent)

# Set CUDA and C++ standards
set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Get required packages
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(yaml-cpp REQUIRED)

# Add stb library
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE external/stb)

# Add gltext library
add_library(gltext INTERFACE)
target_include_directories(gltext INTERFACE external/gltext)

# Add OpenImageDenoise
set(OIDN_ROOT "C:/Program Files/OpenImageDenoise")
include_directories("${OIDN_ROOT}/include")

# Get source files
set(SRC_DIR src)
file(GLOB_RECURSE SOURCES ${SRC_DIR}/*.cu)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Set NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 --std=c++17 --use_fast_math -Xptxas -O3")

# Specify the runtime library to avoid conflicts
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Release>:Release>")

# Link dependencies
include_directories(${PROJECT_NAME} PRIVATE ${GLEW_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${GLEW_LIBRARIES}
    SDL2::SDL2
    SDL2::SDL2main
    GLEW::GLEW
    yaml-cpp::yaml-cpp
    stb
    gltext
    opengl32
    "${OIDN_ROOT}/lib/OpenImageDenoise.lib"
    "${OIDN_ROOT}/lib/OpenImageDenoise_core.lib"
)

# Set output directory
set(DIST_DIR dist)
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${DIST_DIR}
)
