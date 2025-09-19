#include "ptex/Texture.h"

#include <stdlib.h>
#include <stdexcept>

#include <vector>
#include <functional>
#include <cstdint>
#include <random>
#include <cmath>

namespace PTex
{
    Texture::Texture(int width, int height) : m_Width(width), m_Height(height), m_Data(width * height * PTEX_TEXTURE_CHANNELS, 0.0f)
    {
    }

    Texture::~Texture()
    {
    }

    Texture &Texture::setData(const float *data, int size)
    {
        int expectedSize = m_Width * m_Height * PTEX_TEXTURE_CHANNELS;
        if (size != expectedSize)
            throw std::runtime_error("Data size does not match texture size");

        std::copy(data, data + size, m_Data.begin());
        return *this;
    }

    Texture &Texture::gradient(vec4 colA, vec4 colB, float angle)
    {
        float rad = angle * (3.14159265f / 180.0f);
        float dx = std::cos(rad);
        float dy = std::sin(rad);

        for (int y = 0; y < m_Height; ++y)
        {
            for (int x = 0; x < m_Width; ++x)
            {
                // projection normalized to [0, 1]
                float t = (x * dx + y * dy) / (m_Width * std::abs(dx) + m_Height * std::abs(dy));
                t = std::clamp(t, 0.0f, 1.0f);

                int idx = (y * m_Width + x) * PTEX_TEXTURE_CHANNELS;

                // blend each channel
                m_Data[idx + 0] = colA.x * (1.0f - t) + colB.x * t; // R / X
                m_Data[idx + 1] = colA.y * (1.0f - t) + colB.y * t; // G / Y
                m_Data[idx + 2] = colA.z * (1.0f - t) + colB.z * t; // B / Z
                m_Data[idx + 3] = colA.w * (1.0f - t) + colB.w * t; // A / W
            }
        }

        return *this;
    }
}
