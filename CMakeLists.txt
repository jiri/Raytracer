cmake_minimum_required(VERSION 3.13)

project(raytracer LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 12)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Executables
add_executable(${PROJECT_NAME} src/main.cu src/cutil_math.h)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)