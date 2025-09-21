struct TextureNode
{
    int id;
    std::string name;
    std::vector<std::shared_ptr<TextureNode>> inputs;
    std::vector<std::string> inputNames;
    unsigned int color;
    unsigned int highlightColor;
    bool compact = false;
    enum class Type
    {
        Gradient,
        Voronoi,
        Noise,
        Mix,
        Blur,
        Color,
        Addition,
        Subtraction,
        Multiplication,
        Division,
        Output
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

    std::string getNameFromType(Type type)
    {
        switch (type)
        {
        case Type::Blur:
            return "Blur";
        case Type::Gradient:
            return "Gradient Texture";
        case Type::Voronoi:
            return "Voronoi Noise";
        case Type::Noise:
            return "Noise";
        case Type::Mix:
            return "Mix";
        case Type::Addition:
            return "Math (Addition)";
        case Type::Subtraction:
            return "Math (Subtraction)";
        case Type::Multiplication:
            return "Math (Multiplication)";
        case Type::Division:
            return "Math (Division)";
        case Type::Color:
            return "Solid Color";
        case Type::Output:
            return "Output";
        default:
            return "NOT IMPLEMENTED IN getNameFromType IN NODES.H";
        }
    }

    unsigned int getColorFromType(Type type)
    {
        switch (type)
        {
        case Type::Blur:
            return IM_COL32(180, 100, 255, 255); // purple
        case Type::Gradient:
            return IM_COL32(255, 140, 0, 255); // orange
        case Type::Voronoi:
            return IM_COL32(0, 200, 120, 255); // teal/green
        case Type::Noise:
            return IM_COL32(200, 200, 200, 255); // light gray
        case Type::Mix:
            return IM_COL32(100, 150, 255, 255); // blue
        case Type::Color:
            return IM_COL32(255, 80, 80, 255); // red
        case Type::Output:
            return IM_COL32(255, 215, 0, 255); // gold/yellow
        default:
            return IM_COL32(80, 80, 80, 255); // fallback gray
        }
    }

    unsigned int getHighlightColor(unsigned int color, float factor = 1.2f)
    {
        unsigned char r = (color & 0xFF);
        unsigned char g = (color >> 8) & 0xFF;
        unsigned char b = (color >> 16) & 0xFF;
        unsigned char a = (color >> 24) & 0xFF;

        r = static_cast<unsigned char>(std::min(255.0f, r * factor));
        g = static_cast<unsigned char>(std::min(255.0f, g * factor));
        b = static_cast<unsigned char>(std::min(255.0f, b * factor));

        return IM_COL32(r, g, b, a);
    }

    TextureNode(int _id, Type _type)
        : id(_id), name(getNameFromType(_type)), color(getColorFromType(_type)), highlightColor(getHighlightColor(getColorFromType(_type))), type(_type)
    {
        switch (type)
        {
        case Type::Mix:
            inputNames = {"Color A", "Color B", "Value"};
            break;
        case Type::Addition:
        case Type::Subtraction:
        case Type::Multiplication:
        case Type::Division:
            inputNames = {"Color A", "Color B"};
            break;
        case Type::Blur:
            inputNames = {"Color"};
            break;
        case Type::Output:
            inputNames = {"Output"};
            break;
        case Type::Color:
            compact = true;
            break;
        }
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
            changed |= ImGui::ColorEdit3(("Color A##" + std::to_string(id)).c_str(), (float *)&colA);
            ImGui::PopItemWidth();
            if (changed)
                params.colorA = PTex::vec4(colA.x, colA.y, colA.z, colA.w);

            ImVec4 colB(params.colorB.x, params.colorB.y, params.colorB.z, params.colorB.w);
            ImGui::PushItemWidth(140);
            changed |= ImGui::ColorEdit3(("Color B##" + std::to_string(id)).c_str(), (float *)&colB);
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
            changed |= ImGui::ColorEdit3(("Color##" + std::to_string(id)).c_str(), (float *)&col);
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

        case Type::Blur:
            ImGui::PushItemWidth(80);
            changed |= ImGui::DragFloat(("Radius##" + std::to_string(id)).c_str(), &params.radius, 0.1f, 0.0f, 100.0f);
            ImGui::PopItemWidth();
            break;

        case Type::Addition:
        case Type::Subtraction:
        case Type::Multiplication:
        case Type::Division:
        {
            Type displayedTypes[] = {Type::Addition, Type::Subtraction, Type::Multiplication, Type::Division};
            const char *typeNames[] = {"Addition", "Subtraction", "Multiplication", "Division"};

            int currentIndex = 0;
            for (int i = 0; i < IM_ARRAYSIZE(displayedTypes); i++)
            {
                if (type == displayedTypes[i])
                {
                    currentIndex = i;
                    break;
                }
            }

            int newIndex = currentIndex;
            ImGui::PushID(id);
            ImGui::PushItemWidth(140);
            if (ImGui::BeginCombo("Operation", typeNames[currentIndex]))
            {
                for (int i = 0; i < IM_ARRAYSIZE(displayedTypes); i++)
                {
                    bool isSelected = (currentIndex == i);
                    if (ImGui::Selectable(typeNames[i], isSelected))
                        newIndex = i;
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::PopItemWidth();
            ImGui::PopID();

            if (newIndex != currentIndex)
            {
                type = displayedTypes[newIndex];
                name = getNameFromType(type);
            }
        }
        case Type::Mix:
            break;
        }

        // Output Pin
        if (type == Type::Output)
        {
            /*ImNodes::BeginOutputAttribute(getOutputPinID(0));
            ImGui::TextDisabled("End");              // Disable Output Pin
            ImNodes::EndOutputAttribute();*/
        }
        else
        {
            ImNodes::BeginOutputAttribute(getOutputPinID(0));
            ImGui::Text("Output");
            ImNodes::EndOutputAttribute();
        }

        return changed;
    }

    PTex::Texture &evaluate()
    {
        std::vector<PTex::Texture *> inputTextures;
        for (auto &in : inputs)
        {
            if (in)
                inputTextures.push_back(&in->evaluate());
        }

        if (type == Type::Output)
        {
            if (!inputs.empty())
                texture = inputs[0]->evaluate();
            else
                texture.zero();
            texture.end();
            return texture;
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
        case Type::Addition:
            if (inputTextures.size() >= 2)
            {
                texture = *inputTextures[0];
                texture.add(*inputTextures[1]);
            }
            else if (inputTextures.size() == 1)
            {
                texture = *inputTextures[0];
            }
            break;
        case Type::Subtraction:
            if (inputTextures.size() >= 2)
            {
                texture = *inputTextures[0];
                texture.sub(*inputTextures[1]);
            }
            else if (inputTextures.size() == 1)
            {
                texture = *inputTextures[0];
            }
            break;
        case Type::Multiplication:
            if (inputTextures.size() >= 2)
            {
                texture = *inputTextures[0];
                texture.multi(*inputTextures[1]);
            }
            else if (inputTextures.size() == 1)
            {
                texture = *inputTextures[0];
            }
            break;
        case Type::Division:
            if (inputTextures.size() >= 2)
            {
                texture = *inputTextures[0];
                texture.divide(*inputTextures[1]);
            }
            else if (inputTextures.size() == 1)
            {
                texture = *inputTextures[0];
            }
            break;
        }

        texture.end();
        return texture;
    }
};