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
# Options
# --------------------------
option(PTEX_USE_OPENGL "Enable OpenGL support in ptexLib" ON)

# --------------------------
# Externals
# --------------------------
add_subdirectory(external)

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

if (PTEX_USE_OPENGL)
    message(STATUS "Building ptexLib with OpenGL support")

    target_link_libraries(ptexLib PUBLIC glad)
    target_compile_definitions(ptexLib PUBLIC PTEX_USE_OPENGL=1)
endif()


# --------------------------
# Demo Executable
# --------------------------

file(GLOB_RECURSE EDITOR_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/editor/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/editor/src/*.cu"
)

add_executable(ptexEditor ${EDITOR_SOURCES})

target_include_directories(ptexEditor
    PRIVATE
    ${CMAKE_SOURCE_DIR}/editor/src
)

target_link_libraries(ptexEditor PRIVATE ptexLib externals)
