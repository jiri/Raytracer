cmake_minimum_required(VERSION 3.13)

project(raytracer LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Executables
add_executable(${PROJECT_NAME} main.cu cutil_math.h)