#pragma once

#include "ptex/Texture.cuh"
#include "ptex/Math.cuh"
#include "ptex/Vector.cuh"

#define CUDA_CHECK(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            printf("CUDA Error: %s\n", std::string(cudaGetErrorString(err)).c_str());        \
            fflush(stdout);                                                                  \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        }                                                                                    \
    } while (0)

namespace PTex
{
    __device__ float d_hash(int x, int y)
    {
        uint32_t h = x * 374761393 + y * 668265263;
        h = (h ^ (h >> 13)) * 1274126177;
        return (h & 0x7fffffff) / float(0x7fffffff);
    }

    __device__ float d_lerp(float a, float b, float t)
    {
        return a + t * (b - a);
    }

    __device__ float d_smoothstep(float t)
    {
        return t * t * (3 - 2 * t);
    }

    __device__ float d_valueNoise(float x, float y)
    {
        int xi = int(floorf(x));
        int yi = int(floorf(y));
        float xf = x - xi;
        float yf = y - yi;

        float v00 = d_hash(xi, yi);
        float v10 = d_hash(xi + 1, yi);
        float v01 = d_hash(xi, yi + 1);
        float v11 = d_hash(xi + 1, yi + 1);

        float u = d_smoothstep(xf);
        float v = d_smoothstep(yf);

        return d_lerp(d_lerp(v00, v10, u), d_lerp(v01, v11, u), v);
    }

    __device__ vec2 d_randomOffset(int cx, int cy)
    {
        float r1 = d_hash(cx, cy);
        float r2 = d_hash(cy, cx);
        return vec2(r1, r2);
    }

    __global__ void gradientKernel(float *data, int width, int height, vec4 colA, vec4 colB, float angle)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        float rad = angle * (3.14159265f / 180.0f);
        vec2 direction(cosf(rad), sinf(rad));

        vec2 pos = vec2(float(x) / float(width), float(y) / float(height));

        float t = pos.x * direction.x + pos.y * direction.y;
        t = fmaxf(0.0f, fminf(1.0f, t));

        vec4 result = colA.lerp(colB, t);

        int idx = (y * width + x) * PTEX_TEXTURE_CHANNELS;
        data[idx + 0] = result.x;
        data[idx + 1] = result.y;
        data[idx + 2] = result.z;
        data[idx + 3] = result.w;
    }

    __global__ void noiseKernel(float *data, int width, int height,
                                float scale, float detail, float roughness, float lacunarity, float distortion)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        vec2 coord(float(x) / float(width), float(y) / float(height));

        if (distortion > 0.0f)
        {
            vec2 distortCoord = coord * 4.0f;
            float dx = d_valueNoise(distortCoord.x, distortCoord.y) * distortion;
            float dy = d_valueNoise(distortCoord.x + 100.0f, distortCoord.y + 100.0f) * distortion;
            coord = coord + vec2(dx, dy);
        }

        float frequency = scale;
        float amplitude = 1.0f;
        float total = 0.0f;
        float maxValue = 0.0f;

        int octaves = int(floorf(detail));
        float frac = detail - octaves;

        for (int i = 0; i < octaves; ++i)
        {
            vec2 noiseCoord = coord * frequency;
            total += d_valueNoise(noiseCoord.x, noiseCoord.y) * amplitude;
            maxValue += amplitude;
            frequency *= lacunarity;
            amplitude *= roughness;
        }

        if (frac > 0.0f)
        {
            vec2 noiseCoord = coord * frequency;
            total += d_valueNoise(noiseCoord.x, noiseCoord.y) * amplitude * frac;
            maxValue += amplitude * frac;
        }

        float noiseVal = total / maxValue;

        int idx = (y * width + x) * PTEX_TEXTURE_CHANNELS;
        data[idx + 0] = noiseVal;
        data[idx + 1] = noiseVal;
        data[idx + 2] = noiseVal;
        data[idx + 3] = 1.0f;
    }

    __global__ void voronoiKernel(float *data, int width, int height,
                                  float scale, float detail, float roughness, float lacunarity, float smoothness)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        vec2 coord(float(x) / float(width), float(y) / float(height));

        float frequency = scale;
        float amplitude = 1.0f;
        float total = 0.0f;
        float maxValue = 0.0f;

        int octaves = int(floorf(detail));
        float frac = detail - octaves;

        for (int i = 0; i < octaves; ++i)
        {
            vec2 scaledCoord = coord * frequency;

            int cellX = int(floorf(scaledCoord.x));
            int cellY = int(floorf(scaledCoord.y));
            float minDist = 1e9f;

            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    vec2 cellPos(float(cellX + dx), float(cellY + dy));
                    vec2 offset = d_randomOffset(cellX + dx, cellY + dy);
                    vec2 pointPos = cellPos + offset;

                    vec2 diff = pointPos - scaledCoord;
                    float dist = diff.length();

                    if (dist < minDist)
                        minDist = dist;
                }
            }

            float val = powf(fmaxf(0.0f, fminf(1.0f, minDist)), smoothness);
            total += val * amplitude;
            maxValue += amplitude;

            frequency *= lacunarity;
            amplitude *= roughness;
        }

        if (frac > 0.0f)
        {
            vec2 scaledCoord = coord * frequency;

            int cellX = int(floorf(scaledCoord.x));
            int cellY = int(floorf(scaledCoord.y));
            float minDist = 1e9f;

            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    vec2 cellPos(float(cellX + dx), float(cellY + dy));
                    vec2 offset = d_randomOffset(cellX + dx, cellY + dy);
                    vec2 pointPos = cellPos + offset;

                    vec2 diff = pointPos - scaledCoord;
                    float dist = diff.length();

                    if (dist < minDist)
                        minDist = dist;
                }
            }

            float val = powf(fmaxf(0.0f, fminf(1.0f, minDist)), smoothness);
            total += val * amplitude * frac;
            maxValue += amplitude * frac;
        }

        float noiseVal = total / maxValue;

        int idx = (y * width + x) * PTEX_TEXTURE_CHANNELS;
        data[idx + 0] = noiseVal;
        data[idx + 1] = noiseVal;
        data[idx + 2] = noiseVal;
        data[idx + 3] = 1.0f;
    }

    template <typename Op>
    __global__ void mathFunctionKernel(float *a_data, const float *b_data,
                                       int a_width, int a_height,
                                       int b_width, int b_height, Op op)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= a_width || y >= a_height)
            return;

        vec2 aUV(float(x) / float(a_width), float(y) / float(a_height));
        int aIdx = (y * a_width + x) * PTEX_TEXTURE_CHANNELS;

        vec2 bUV(aUV.x * (b_width - 1), aUV.y * (b_height - 1));
        int sx = int(bUV.x);
        int sy = int(bUV.y);
        int bIdx = (sy * b_width + sx) * PTEX_TEXTURE_CHANNELS;

        vec4 aColor(a_data[aIdx], a_data[aIdx + 1], a_data[aIdx + 2], a_data[aIdx + 3]);
        vec4 bColor(b_data[bIdx], b_data[bIdx + 1], b_data[bIdx + 2], b_data[bIdx + 3]);

        vec4 result = op(aColor, bColor);

        // Store result
        a_data[aIdx + 0] = result.x;
        a_data[aIdx + 1] = result.y;
        a_data[aIdx + 2] = result.z;
        a_data[aIdx + 3] = result.w;
    }

    __global__ void mixKernel(float *data, const float *value_data, const float *source_data,
                              int width, int height, int value_width, int value_height,
                              int source_width, int source_height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        vec2 uv(float(x) / float(width), float(y) / float(height));

        // Sample value texture
        vec2 valueCoord(uv.x * (value_width - 1), uv.y * (value_height - 1));
        int vx = int(valueCoord.x);
        int vy = int(valueCoord.y);
        int vIdx = (vy * value_width + vx) * PTEX_TEXTURE_CHANNELS;
        float blend = fmaxf(0.0f, fminf(1.0f, value_data[vIdx]));

        // Sample source texture
        vec2 sourceCoord(uv.x * (source_width - 1), uv.y * (source_height - 1));
        int sx = int(sourceCoord.x);
        int sy = int(sourceCoord.y);
        int sIdx = (sy * source_width + sx) * PTEX_TEXTURE_CHANNELS;

        int idx = (y * width + x) * PTEX_TEXTURE_CHANNELS;

        // Mix colors using vec4
        vec4 baseColor(data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);
        vec4 sourceColor(source_data[sIdx], source_data[sIdx + 1],
                         source_data[sIdx + 2], source_data[sIdx + 3]);

        vec4 result = baseColor.lerp(sourceColor, blend);

        data[idx + 0] = result.x;
        data[idx + 1] = result.y;
        data[idx + 2] = result.z;
        data[idx + 3] = result.w;
    }

    __global__ void grayscaleKernel(float *data, int width, int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int idx = (y * width + x) * PTEX_TEXTURE_CHANNELS;

        vec4 color(data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);

        // Luminance weights as vec4
        vec4 weights(0.2126f, 0.7152f, 0.0722f, 0.0f);
        float gray = color.x * weights.x + color.y * weights.y + color.z * weights.z;

        data[idx + 0] = gray;
        data[idx + 1] = gray;
        data[idx + 2] = gray;
        data[idx + 3] = color.w; // Keep alpha
    }

    __global__ void blurHorizontalKernel(float *output, const float *input, const float *kernel,
                                         int width, int height, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int half = kernelSize / 2;
        vec4 result(0.0f, 0.0f, 0.0f, 0.0f);

        for (int k = -half; k <= half; ++k)
        {
            int sx = max(0, min(width - 1, x + k));
            int srcIdx = (y * width + sx) * PTEX_TEXTURE_CHANNELS;

            vec4 sample(input[srcIdx], input[srcIdx + 1], input[srcIdx + 2], input[srcIdx + 3]);
            float weight = kernel[k + half];

            result = result + sample * weight;
        }

        int outIdx = (y * width + x) * PTEX_TEXTURE_CHANNELS;
        output[outIdx + 0] = result.x;
        output[outIdx + 1] = result.y;
        output[outIdx + 2] = result.z;
        output[outIdx + 3] = result.w;
    }

    __global__ void blurVerticalKernel(float *output, const float *input, const float *kernel,
                                       int width, int height, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int half = kernelSize / 2;
        vec4 result(0.0f, 0.0f, 0.0f, 0.0f);

        for (int k = -half; k <= half; ++k)
        {
            int sy = max(0, min(height - 1, y + k));
            int srcIdx = (sy * width + x) * PTEX_TEXTURE_CHANNELS;

            vec4 sample(input[srcIdx], input[srcIdx + 1], input[srcIdx + 2], input[srcIdx + 3]);
            float weight = kernel[k + half];

            result = result + sample * weight;
        }

        int outIdx = (y * width + x) * PTEX_TEXTURE_CHANNELS;
        output[outIdx + 0] = result.x;
        output[outIdx + 1] = result.y;
        output[outIdx + 2] = result.z;
        output[outIdx + 3] = result.w;
    }
}