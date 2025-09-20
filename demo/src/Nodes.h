struct TextureNode
{
    int id;
    std::string name;
    std::vector<std::shared_ptr<TextureNode>> inputs;
    std::vector<std::string> inputNames;
    enum class Type
    {
        Gradient,
        Voronoi,
        Noise,
        Mix,
        Blur,
        Color
    } type;
    struct Params
    {
        PTex::vec4 colorA = PTex::vec4(1.0f);
        PTex::vec4 colorB = PTex::vec4(1.0f);
        float angle = 0.0f;
        float scale = 5.0f;
        float radius = 2.0f;
    } params;

    PTex::Texture texture;

    TextureNode(int _id, std::string _name, Type _type)
        : id(_id), name(_name), type(_type)
    {
        // Default input pins if not explicitly passed
        if (type == Type::Mix)
            inputNames = {"Color A", "Color B", "Value"};
        if (type == Type::Blur)
            inputNames = {"Color"};
    }

    int getInputPinID(int index) { return id * 100 + index; }
    int getOutputPinID(int index) { return id * 1000 + index; }

    bool renderAttributes()
    {
        bool changed = false;

        // Input Pins
        for (int i = 0; i < inputNames.size(); i++)
        {
            ImNodes::BeginInputAttribute(getInputPinID(i));
            ImGui::Text("%s", inputNames[i].c_str());
            ImNodes::EndInputAttribute();
        }

        // Input Arguments
        switch (type)
        {
        case Type::Gradient:
        {
            ImVec4 colA(params.colorA.x, params.colorA.y, params.colorA.z, params.colorA.w);
            ImGui::PushItemWidth(140);
            changed |= ImGui::ColorEdit4(("Color A##" + std::to_string(id)).c_str(), (float *)&colA);
            ImGui::PopItemWidth();
            if (changed)
                params.colorA = PTex::vec4(colA.x, colA.y, colA.z, colA.w);

            ImVec4 colB(params.colorB.x, params.colorB.y, params.colorB.z, params.colorB.w);
            ImGui::PushItemWidth(140);
            changed |= ImGui::ColorEdit4(("Color B##" + std::to_string(id)).c_str(), (float *)&colB);
            ImGui::PopItemWidth();
            if (changed)
                params.colorB = PTex::vec4(colB.x, colB.y, colB.z, colB.w);

            ImGui::PushItemWidth(140);
            changed |= ImGui::DragFloat(("Angle##" + std::to_string(id)).c_str(), &params.angle, 1.0f, 0.0f, 360.0f, "%.0f");
            ImGui::PopItemWidth();
            break;
        }
        case Type::Color:
        {
            ImVec4 col(params.colorA.x, params.colorA.y, params.colorA.z, params.colorA.w);
            ImGui::PushItemWidth(140);
            changed |= ImGui::ColorEdit4(("Color##" + std::to_string(id)).c_str(), (float *)&col);
            ImGui::PopItemWidth();
            if (changed)
                params.colorA = PTex::vec4(col.x, col.y, col.z, col.w);
            break;
        }

        case Type::Voronoi:
        case Type::Noise:
            ImGui::PushItemWidth(80);
            changed |= ImGui::DragFloat(("Scale##" + std::to_string(id)).c_str(), &params.scale, 0.01f, 0.0f, 100.0f);
            ImGui::PopItemWidth();
            break;

        case Type::Mix:
            break;

        case Type::Blur:
            ImGui::PushItemWidth(80);
            changed |= ImGui::DragFloat(("Radius##" + std::to_string(id)).c_str(), &params.radius, 0.1f, 0.0f, 100.0f);
            ImGui::PopItemWidth();
            break;
        }

        // Output Pin
        ImNodes::BeginOutputAttribute(getOutputPinID(0));
        ImGui::Text("Output");
        ImNodes::EndOutputAttribute();

        return changed;
    }

    PTex::Texture &evaluate()
    {
        std::vector<PTex::Texture *> inputTextures;
        for (auto &in : inputs)
        {
            inputTextures.push_back(&in->evaluate());
        }

        texture.zero();

        switch (type)
        {
        case Type::Gradient:
            texture.gradient(params.colorA, params.colorB, params.angle);
            break;

        case Type::Voronoi:
            texture.voronoi(params.scale);
            break;

        case Type::Noise:
            texture.noise(params.scale);
            break;

        case Type::Mix:
            if (inputTextures.size() >= 3)
            {
                // Mix into the first input’s texture
                inputTextures[0]->mix(*inputTextures[2], *inputTextures[1]);
                texture = *inputTextures[0];
            }
            else if (inputTextures.size() == 1)
            {
                texture = *inputTextures[0];
            }
            break;

        case Type::Blur:
            if (!inputTextures.empty())
            {
                inputTextures[0]->blur(params.radius);
                texture = *inputTextures[0];
            }
            else
            {
                texture.blur(params.radius);
            }
            break;
        case Type::Color:
            texture.fill(params.colorA);
            break;
        }

        texture.end();
        return texture;
    }
};
