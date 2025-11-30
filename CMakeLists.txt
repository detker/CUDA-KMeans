cmake_minimum_required(VERSION 3.18)
project(KMeans LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDAToolkit_ROOT "/opt/cuda")
find_package(CUDAToolkit REQUIRED)

# NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --extended-lambda \
    --expt-relaxed-constexpr")

# Set path to external libraries
set(EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/cuda
    ${PROJECT_SOURCE_DIR}/include
    ${EXTERNAL_DIR}/glfw/include
    ${EXTERNAL_DIR}/glew/include
)

# Specify library directories
link_directories(
    ${EXTERNAL_DIR}/glfw/lib
    ${EXTERNAL_DIR}/glew/lib
)

# Source files (only .cpp and .cu)
file(GLOB_RECURSE CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_SOURCES  "${PROJECT_SOURCE_DIR}/cuda/*.cu")
file(GLOB_RECURSE C_SOURCES  "${PROJECT_SOURCE_DIR}/src/*.c")

# Optional: list headers for IDE visibility
file(GLOB_RECURSE HEADER_FILES "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/cuda/*.cuh")

add_executable(${PROJECT_NAME} ${C_SOURCES} ${CPP_SOURCES} ${CU_SOURCES} ${HEADER_FILES})

# Find OpenGL (system library)
find_package(OpenGL REQUIRED)

# Add external library include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${EXTERNAL_DIR}/glfw/include
    ${EXTERNAL_DIR}/glew/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries with full paths from external directory
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    ${EXTERNAL_DIR}/glfw/lib/libglfw.so.3.3
    ${EXTERNAL_DIR}/glew/lib/libGLEW.so.2.2.0
    dl
    pthread
    X11
    Xxf86vm
    Xrandr
    Xi
)

# CUDA separable compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)