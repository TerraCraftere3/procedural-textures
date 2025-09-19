#pragma once

#include <vector>

#include "Core.h"
#include "Vector.h"

#define PTEX_TEXTURE_CHANNELS 4

namespace PTex
{

    class PTEX_API Texture
    {
    public:
        Texture(int width = 256, int height = 256);
        ~Texture();

        int width() const { return m_Width; }
        int height() const { return m_Height; }

        const float *data() const { return m_Data.data(); }
        float *data() { return m_Data.data(); }

        Texture &setData(const float *data, int size);
        Texture &gradient(vec4 colA, vec4 colB, float angle = 0.0f);

    private:
        int m_Width, m_Height;
        std::vector<float> m_Data;
    };

}