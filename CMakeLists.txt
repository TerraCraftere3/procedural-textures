cmake_minimum_required(VERSION 3.20)

# --------------------------
# Project
# --------------------------
project(ptexProject LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------------
# Options
# --------------------------
option(PTEX_BUILD_SHARED "Build ptexLib as a shared library (DLL/.so)" ON)


# --------------------------
# Library
# --------------------------
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/lib/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/lib/src/*.c"
)

if(PTEX_BUILD_SHARED)
    add_library(ptexLib SHARED ${LIB_SOURCES})
    target_compile_definitions(ptexLib PRIVATE PTEX_BUILD_DLL)
    target_compile_definitions(ptexLib INTERFACE PTEX_USE_DLL)
else()
    add_library(ptexLib STATIC ${LIB_SOURCES})
endif()

target_include_directories(ptexLib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/lib/include
)

# --------------------------
# Demo Executable
# --------------------------
file(GLOB_RECURSE DEMO_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/demo/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/demo/src/*.c"
)

add_executable(ptexDemo ${DEMO_SOURCES})

target_include_directories(ptexDemo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/demo/src
)

target_link_libraries(ptexDemo PRIVATE ptexLib)

# --------------------------
# Warnings
# --------------------------
if (MSVC)
    target_compile_options(ptexLib PRIVATE /W4 /permissive-)
    target_compile_options(ptexDemo PRIVATE /W4 /permissive-)
else()
    target_compile_options(ptexLib PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(ptexDemo PRIVATE -Wall -Wextra -Wpedantic)
endif()
