cmake_minimum_required(VERSION 3.20)
project(external)

# --------------------------
# GLFW
# --------------------------
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
set(GLFW_INSTALL OFF CACHE INTERNAL "")
add_subdirectory(glfw)

# --------------------------
# GLAD
# --------------------------
set(GLAD_SOURCES
    glad/src/glad.c
)
add_library(glad STATIC ${GLAD_SOURCES})
target_include_directories(glad PUBLIC glad/include)

# --------------------------
# IMGUI
# --------------------------
set(IMGUI_SOURCES
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC
    imgui
    imgui/backends
)
# Link ImGui to GLFW + OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# --------------------------
# Convenience target
# --------------------------
add_library(externals INTERFACE)
target_link_libraries(externals INTERFACE glad glfw imgui)
