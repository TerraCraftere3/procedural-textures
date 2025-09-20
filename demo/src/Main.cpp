#include <ptex/PTex.h>

/*#include <fstream>
#include <algorithm>
#include <chrono>
#include <iostream>

int main()
{
    using namespace PTex;

    printf("Starting Texture Creation\n");
    auto start = std::chrono::high_resolution_clock::now();

    int size = 512;
#define SIZED_TEXTURE() Texture(size, size)
    Texture tex(size, size);
    tex.gradient(vec4(1.0f, 0.3f, 0.2f, 1.0f), vec4(0.2f, 0.3f, 1.0f, 1.0f), 45.0f)
        .mix(SIZED_TEXTURE().voronoi(), SIZED_TEXTURE().noise())
        .blur(25.0f)
        .multi(SIZED_TEXTURE().voronoi())
        .copy();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Texture creation took %.3f ms\n", elapsed.count());
    writePPM("output.ppm", tex);
    printf("Wrote file \"output.ppm\"...\n");
    return 0;
}*/

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <iostream>
#include <chrono>

int main()
{
    using namespace PTex;

    // -------------------
    // Initialize GLFW
    // -------------------
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(1280, 720, "Procedural Textures", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync

    // -------------------
    // Load OpenGL using GLAD
    // -------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    // -------------------
    // Setup ImGui
    // -------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    // Enable docking
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // -------------------
    // Texture Creation
    // -------------------
    printf("Starting Texture Creation\n");
    auto start = std::chrono::high_resolution_clock::now();

    int size = 512;
#define SIZED_TEXTURE() Texture(size, size)
    Texture tex(size, size);
    tex.gradient(vec4(1.0f, 0.3f, 0.2f, 1.0f), vec4(0.2f, 0.3f, 1.0f, 1.0f), 45.0f)
        .mix(SIZED_TEXTURE().voronoi(), SIZED_TEXTURE().noise())
        .blur(25.0f)
        .multi(SIZED_TEXTURE().voronoi())
        .end();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Texture creation took %.3f ms\n", elapsed.count());
    fflush(stdout);

    // -------------------
    // Main loop
    // -------------------
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Dockspace in a single window
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar |
                                        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_NoNavFocus;

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        ImGui::SetNextWindowSize(ImVec2((float)display_w, (float)display_h));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::Begin("Dockspace", nullptr, window_flags);
        ImGuiID dockspace_id = ImGui::GetID("Dockspace");
        ImGui::DockSpace(dockspace_id, ImVec2(0, 0), ImGuiDockNodeFlags_None);
        ImGui::End();

        ImGui::PopStyleVar();

        ImGui::ShowDemoWindow();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::Begin("Viewport");
        {
            int texWidth = tex.width();
            int texHeight = tex.height();

            ImVec2 availSize = ImGui::GetContentRegionAvail();
            float aspect = (float)texWidth / (float)texHeight;
            ImVec2 imageSize = availSize;

            if (imageSize.x / aspect <= imageSize.y)
            {
                imageSize.y = imageSize.x / aspect;
            }
            else
            {
                imageSize.x = imageSize.y * aspect;
            }

            ImGui::Image(tex.getTextureID(), imageSize);
        }
        ImGui::End();

        ImGui::PopStyleVar();

        // Render
        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // -------------------
    // Cleanup
    // -------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
