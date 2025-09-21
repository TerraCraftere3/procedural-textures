#include <ptex/PTex.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imnodes.h"
#include <iostream>
#include <chrono>
#include <vector>

#include "Nodes.h"

bool showTextureNodeEditor(std::vector<std::shared_ptr<TextureNode>> &nodes)
{
    ImNodes::BeginNodeEditor();

    bool changed = false;

    for (auto &node : nodes)
    {
        ImNodes::PushColorStyle(
            ImNodesCol_TitleBar, node->color);
        ImNodes::PushColorStyle(
            ImNodesCol_TitleBarHovered, node->highlightColor);
        ImNodes::PushColorStyle(
            ImNodesCol_TitleBarSelected, node->highlightColor);
        ImNodes::BeginNode(node->id);

        ImNodes::BeginNodeTitleBar();
        ImGui::Text("%s", node->name.c_str());
        ImNodes::EndNodeTitleBar();

        changed |= node->renderAttributes();

        // Display texture
        if (node->texture.width() > 0)
            ImGui::Image((void *)(intptr_t)node->texture.getTextureID(), ImVec2(128, 128));

        ImNodes::EndNode();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }

    for (auto &node : nodes)
    {
        for (int i = 0; i < node->inputs.size(); i++)
        {
            auto inputNode = node->inputs[i];
            ImNodes::Link(node->id * 100 + i,
                          inputNode->getOutputPinID(0),
                          node->getInputPinID(i));
        }
    }

    static bool spawnNode = false;
    static TextureNode::Type spawnType;
    static ImVec2 spawnPos;

    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup) &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Right))
    {
        ImGui::OpenPopup("node_editor_context");
    }

    if (ImGui::BeginPopup("node_editor_context"))
    {
        ImVec2 clickPos = ImGui::GetMousePos();

        ImGui::SeparatorText("Spawn Node");
        if (ImGui::MenuItem("Add Gradient Node"))
        {
            spawnNode = true;
            spawnType = TextureNode::Type::Gradient;
        }
        if (ImGui::MenuItem("Add Voronoi Node"))
        {
            spawnNode = true;
            spawnType = TextureNode::Type::Voronoi;
        }
        if (ImGui::MenuItem("Add Noise Node"))
        {
            spawnNode = true;
            spawnType = TextureNode::Type::Voronoi;
        }
        if (ImGui::MenuItem("Add Mix Node"))
        {
            spawnNode = true;
            spawnType = TextureNode::Type::Mix;
        }
        if (ImGui::MenuItem("Add Blur Node"))
        {
            spawnNode = true;
            spawnType = TextureNode::Type::Blur;
        }
        if (ImGui::MenuItem("Add Color Node"))
        {
            spawnNode = true;
            spawnType = TextureNode::Type::Color;
        }

        spawnPos = clickPos;

        ImGui::EndPopup();
    }

    ImNodes::EndNodeEditor();

    // Link creation
    int start_attr, end_attr;
    if (ImNodes::IsLinkCreated(&start_attr, &end_attr))
    {
        for (auto &node : nodes)
        {
            for (int i = 0; i < node->inputNames.size(); i++)
            {
                if (node->getInputPinID(i) == end_attr)
                {
                    if (node->inputs.size() <= i)
                        node->inputs.resize(i + 1);

                    for (auto &other : nodes)
                    {
                        if (other->getOutputPinID(0) == start_attr)
                        {
                            node->inputs[i] = other; // set / replace
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    // link destruction
    int destroyed_link_id;
    if (ImNodes::IsLinkDestroyed(&destroyed_link_id))
    {
        for (auto &node : nodes)
        {
            for (int i = 0; i < node->inputs.size(); i++)
            {
                int expected_id = node->id * 100 + i;
                if (expected_id == destroyed_link_id)
                {
                    node->inputs[i].reset(); // remove connection
                    changed = true;
                }
            }
        }
    }

    if (spawnNode)
    {
        int newId = (int)nodes.size() + 1;

        auto newNode = std::make_shared<TextureNode>(newId, spawnType);
        nodes.push_back(newNode);

        ImNodes::SetNodeEditorSpacePos(newId, spawnPos);

        spawnNode = false;
    }
    return changed;
}

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
    glfwMaximizeWindow(window);

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
    ImNodes::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    // Enable docking
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImNodes::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // -------------------
    // Texture Creation
    // -------------------
    std::vector<std::shared_ptr<TextureNode>> nodes;

    nodes.push_back(std::make_shared<TextureNode>(0, TextureNode::Type::Output));
    nodes.push_back(std::make_shared<TextureNode>(1, TextureNode::Type::Voronoi));
    nodes.push_back(std::make_shared<TextureNode>(2, TextureNode::Type::Gradient));
    nodes.push_back(std::make_shared<TextureNode>(3, TextureNode::Type::Color));
    nodes.push_back(std::make_shared<TextureNode>(4, TextureNode::Type::Mix));

    nodes[2]->params.colorA = vec4(1.0f, 0.3f, 0.2f, 1.0f); // Gradient Node
    nodes[2]->params.colorB = vec4(2.0f, 0.3f, 1.0f, 1.0f); //

    nodes[3]->params.colorA = vec4(0.0f, 0.0f, 0.0f, 1.0f); // Color Node

    nodes[4]->inputs.push_back(nodes[3]); //
    nodes[4]->inputs.push_back(nodes[2]); // Mix Node
    nodes[4]->inputs.push_back(nodes[1]); //

    nodes[0]->inputs.push_back(nodes[4]); // Output Node

    // -------------------
    // Main loop
    // -------------------

    constexpr int MAX_FRAMES = 100;     // Number of points in the graph
    float renderTimes[MAX_FRAMES] = {}; // Circular buffer
    int frameIndex = 0;

    float renderTime = 0.0f;
    bool bakeTexture = true;
    bool alwaysRender = false;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Dockspace setup...
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

        // Texture Editor
        ImGui::Begin("Texture Editor");
        {
            ImGui::Text("Render Time: %.3f ms", renderTime);
            ImGui::Checkbox("Rerender every Frame", &alwaysRender);
            if (alwaysRender)
                ImGui::PlotLines("Render Times", renderTimes, MAX_FRAMES, frameIndex, "ms", 0.0f, 50.0f, ImVec2(0, 80));

            bakeTexture |= showTextureNodeEditor(nodes);
            if (bakeTexture || alwaysRender)
            {
                auto start = std::chrono::high_resolution_clock::now();
                nodes.front()->evaluate();
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::milli> elapsed = end - start;
                renderTime = elapsed.count();
                bakeTexture = false;

                // Store render time in the circular buffer
                renderTimes[frameIndex] = renderTime;
                frameIndex = (frameIndex + 1) % MAX_FRAMES;
            }
        }
        ImGui::End();

        // Viewport rendering...
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("Viewport");
        {
            PTex::Texture &finalTex = nodes.front()->texture;
            int texWidth = finalTex.width();
            int texHeight = finalTex.height();

            ImVec2 availSize = ImGui::GetContentRegionAvail();
            float aspect = (float)texWidth / (float)texHeight;
            ImVec2 imageSize = availSize;

            if (imageSize.x / aspect <= imageSize.y)
                imageSize.y = imageSize.x / aspect;
            else
                imageSize.x = imageSize.y * aspect;

            ImGui::Image(finalTex.getTextureID(), imageSize);
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
    ImNodes::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
