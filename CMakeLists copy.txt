cmake_minimum_required(VERSION 3.18)
project(KMeans LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --compiler-bindir=/usr/bin/g++-9 \
    --extended-lambda \
    --expt-relaxed-constexpr")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/cuda
    ${PROJECT_SOURCE_DIR}/include
)

# Source files (only .cpp and .cu)
file(GLOB_RECURSE CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_SOURCES  "${PROJECT_SOURCE_DIR}/cuda/*.cu")
file(GLOB_RECURSE C_SOURCES  "${PROJECT_SOURCE_DIR}/src/*.c")

# Optional: list headers for IDE visibility
file(GLOB_RECURSE HEADER_FILES "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/cuda/*.cuh")

add_executable(${PROJECT_NAME} ${C_SOURCES} ${CPP_SOURCES} ${CU_SOURCES} ${HEADER_FILES})

# Libraries (example: OpenGL, GLFW, GLEW)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    glfw
    GLEW::GLEW
)

# CUDA separable compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)
