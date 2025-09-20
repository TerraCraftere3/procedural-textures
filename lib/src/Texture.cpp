#include "ptex/Texture.h"

#include <stdlib.h>
#include <stdexcept>

#include <vector>
#include <functional>
#include <cstdint>
#include <random>
#include <cmath>

static float hash(int x, int y)
{
    // Integer hash to [0,1]
    uint32_t h = x * 374761393 + y * 668265263; // Large primes
    h = (h ^ (h >> 13)) * 1274126177;
    return (h & 0x7fffffff) / float(0x7fffffff);
}

static float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

static float smoothstep(float t)
{
    return t * t * (3 - 2 * t);
}

static float valueNoise(float x, float y)
{
    int xi = int(std::floor(x));
    int yi = int(std::floor(y));
    float xf = x - xi;
    float yf = y - yi;

    float v00 = hash(xi, yi);
    float v10 = hash(xi + 1, yi);
    float v01 = hash(xi, yi + 1);
    float v11 = hash(xi + 1, yi + 1);

    float u = smoothstep(xf);
    float v = smoothstep(yf);

    return lerp(lerp(v00, v10, u), lerp(v01, v11, u), v);
}

static void randomOffset(int cx, int cy, float &ox, float &oy)
{
    float r1 = hash(cx, cy);
    float r2 = hash(cy, cx);
    ox = r1; // [0,1]
    oy = r2; // [0,1]
}

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

    Texture &Texture::noise(float scale, float detail, float roughness, float lacunarity, float distortion)
    {
        for (int y = 0; y < m_Height; ++y)
        {
            for (int x = 0; x < m_Width; ++x)
            {
                float nx = float(x) / float(m_Width);
                float ny = float(y) / float(m_Height);

                // Apply distortion by warping coordinates
                if (distortion > 0.0f)
                {
                    float dx = valueNoise(nx * 4.0f, ny * 4.0f) * distortion;
                    float dy = valueNoise(nx * 4.0f + 100.0f, ny * 4.0f + 100.0f) * distortion;
                    nx += dx;
                    ny += dy;
                }

                float frequency = scale;
                float amplitude = 1.0f;
                float total = 0.0f;
                float maxValue = 0.0f;

                int octaves = (int)std::floor(detail);
                float frac = detail - octaves; // fractional octave blend

                // integer octaves
                for (int i = 0; i < octaves; ++i)
                {
                    total += valueNoise(nx * frequency, ny * frequency) * amplitude;
                    maxValue += amplitude;

                    frequency *= lacunarity;
                    amplitude *= roughness;
                }

                // fractional octave
                if (frac > 0.0f)
                {
                    total += valueNoise(nx * frequency, ny * frequency) * amplitude * frac;
                    maxValue += amplitude * frac;
                }

                float noiseVal = (total / maxValue);

                int idx = (y * m_Width + x) * PTEX_TEXTURE_CHANNELS;
                m_Data[idx + 0] = noiseVal;
                m_Data[idx + 1] = noiseVal;
                m_Data[idx + 2] = noiseVal;
                m_Data[idx + 3] = 1.0f; // Alpha
            }
        }

        return *this;
    }

    Texture &Texture::voronoi(float scale, float detail, float roughness, float lacunarity, float smoothness)
    {
        for (int y = 0; y < m_Height; ++y)
        {
            for (int x = 0; x < m_Width; ++x)
            {
                float nx = float(x) / float(m_Width);
                float ny = float(y) / float(m_Height);

                float frequency = scale;
                float amplitude = 1.0f;
                float total = 0.0f;
                float maxValue = 0.0f;

                int octaves = (int)std::floor(detail);
                float frac = detail - octaves;

                for (int i = 0; i < octaves; ++i)
                {
                    // Scale into cell space
                    float px = nx * frequency;
                    float py = ny * frequency;

                    int cellX = int(std::floor(px));
                    int cellY = int(std::floor(py));

                    float minDist = 1e9f;

                    // Search neighboring cells
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            int cx = cellX + dx;
                            int cy = cellY + dy;

                            float ox, oy;
                            randomOffset(cx, cy, ox, oy);

                            float fx = (float)cx + ox;
                            float fy = (float)cy + oy;

                            float dxp = fx - px;
                            float dyp = fy - py;
                            float dist = std::sqrt(dxp * dxp + dyp * dyp);

                            if (dist < minDist)
                                minDist = dist;
                        }
                    }

                    // Apply smoothness shaping (controls sharpness of cell edges)
                    float val = std::pow(std::clamp(minDist, 0.0f, 1.0f), smoothness);

                    total += val * amplitude;
                    maxValue += amplitude;

                    frequency *= lacunarity;
                    amplitude *= roughness;
                }

                // Fractional octave blend
                if (frac > 0.0f)
                {
                    float px = nx * frequency;
                    float py = ny * frequency;

                    int cellX = int(std::floor(px));
                    int cellY = int(std::floor(py));
                    float minDist = 1e9f;

                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            int cx = cellX + dx;
                            int cy = cellY + dy;

                            float ox, oy;
                            randomOffset(cx, cy, ox, oy);

                            float fx = (float)cx + ox;
                            float fy = (float)cy + oy;

                            float dxp = fx - px;
                            float dyp = fy - py;
                            float dist = std::sqrt(dxp * dxp + dyp * dyp);

                            if (dist < minDist)
                                minDist = dist;
                        }
                    }

                    float val = std::pow(std::clamp(minDist, 0.0f, 1.0f), smoothness);
                    total += val * amplitude * frac;
                    maxValue += amplitude * frac;
                }

                float noiseVal = total / maxValue;

                int idx = (y * m_Width + x) * PTEX_TEXTURE_CHANNELS;
                m_Data[idx + 0] = noiseVal;
                m_Data[idx + 1] = noiseVal;
                m_Data[idx + 2] = noiseVal;
                m_Data[idx + 3] = 1.0f;
            }
        }

        return *this;
    }

    Texture &Texture::grayscale()
    {
        for (int y = 0; y < m_Height; ++y)
        {
            for (int x = 0; x < m_Width; ++x)
            {
                int idx = (y * m_Width + x) * PTEX_TEXTURE_CHANNELS;

                float r = m_Data[idx + 0];
                float g = m_Data[idx + 1];
                float b = m_Data[idx + 2];
                float a = m_Data[idx + 3];

                // Standard luminance weights (ITU-R BT.709)
                float gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;

                m_Data[idx + 0] = gray;
                m_Data[idx + 1] = gray;
                m_Data[idx + 2] = gray;
                m_Data[idx + 3] = a; // keep alpha
            }
        }

        return *this;
    }
}
