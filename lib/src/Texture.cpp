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

    Texture &Texture::gradient(float angle)
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
                for (int c = 0; c < PTEX_TEXTURE_CHANNELS; ++c)
                    m_Data[idx + c] = t;
            }
        }

        return *this;
    }

}
