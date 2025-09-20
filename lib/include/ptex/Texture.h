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
        Texture &noise(float scale = 5.0f, float detail = 2.0f, float roughness = 0.5f, float lacunarity = 2.0f, float distortion = 0.0f);
        Texture &grayscale();

    private:
        int m_Width,
            m_Height;
        std::vector<float> m_Data;
    };

}