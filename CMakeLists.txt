cmake_minimum_required(VERSION 3.20)

# --------------------------
# Project
# --------------------------
project(ptexProject LANGUAGES C CXX CUDA)

# --------------------------
# Language standards
# --------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# --------------------------
# Library
# --------------------------
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/lib/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/lib/src/*.cu"
)

add_library(ptexLib ${LIB_SOURCES})

target_include_directories(ptexLib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/lib/include
)

# --------------------------
# Demo Executable
# --------------------------
file(GLOB_RECURSE DEMO_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/demo/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/demo/src/*.cu"
)

add_executable(ptexDemo ${DEMO_SOURCES})

target_include_directories(ptexDemo
    PRIVATE
    ${CMAKE_SOURCE_DIR}/demo/src
)

target_link_libraries(ptexDemo PRIVATE ptexLib)
