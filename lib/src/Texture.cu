#include "ptex/Texture.cuh"

#include <iostream>
#include <stdlib.h>
#include <stdexcept>
#include <vector>
#include <functional>
#include <cstdint>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA error checking macro
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
    // ===========================================
    // CUDA KERNEL
    // ===========================================

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

    __global__ void gradientKernelSimple(float *data, int width, int height,
                                         float r1, float g1, float b1, float a1,
                                         float r2, float g2, float b2, float a2,
                                         float angle)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        float rad = angle * (3.14159265f / 180.0f);
        float dirX = cosf(rad);
        float dirY = sinf(rad);

        float posX = float(x) / float(width);
        float posY = float(y) / float(height);

        float t = posX * dirX + posY * dirY;
        t = fmaxf(0.0f, fminf(1.0f, t));

        int idx = (y * width + x) * 4; // Assume 4 channels

        data[idx + 0] = r1 + t * (r2 - r1);
        data[idx + 1] = g1 + t * (g2 - g1);
        data[idx + 2] = b1 + t * (b2 - b1);
        data[idx + 3] = a1 + t * (a2 - a1);
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

    // ===========================================
    // TEXTURE CLASS
    // ===========================================

    Texture::Texture(int width, int height) : m_Width(width), m_Height(height)
    {
        if (width <= 0 || height <= 0)
        {
            throw std::runtime_error("Invalid texture dimensions");
        }

        size_t size = width * height * PTEX_TEXTURE_CHANNELS;
        size_t bytes = size * sizeof(float);

        CUDA_CHECK(cudaMallocHost(&m_Data, bytes));
        memset(m_Data, 0, bytes);

        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMemset(d_data, 0, bytes));

        // Verify the allocation worked
        if (!d_data)
        {
            throw std::runtime_error("Failed to allocate device memory");
        }
    }

    Texture::~Texture()
    {
        if (d_data)
        {
            cudaFree(d_data);
        }
        if (m_Data)
        {
            cudaFreeHost(m_Data);
        }
    }

    Texture &Texture::setData(const float *data, int size)
    {
        int expectedSize = m_Width * m_Height * PTEX_TEXTURE_CHANNELS;
        if (size != expectedSize)
            throw std::runtime_error("Data size does not match texture size");

        size_t bytes = size * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice));
        return *this;
    }

    Texture &Texture::gradient(vec4 colA, vec4 colB, float angle)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        gradientKernel<<<gridSize, blockSize>>>(d_data, m_Width, m_Height, colA, colB, angle);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    Texture &Texture::noise(float scale, float detail, float roughness, float lacunarity, float distortion)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        noiseKernel<<<gridSize, blockSize>>>(d_data, m_Width, m_Height, scale, detail, roughness, lacunarity, distortion);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    Texture &Texture::voronoi(float scale, float detail, float roughness, float lacunarity, float smoothness)
    {
        if (scale <= 0.0f)
            scale = 1.0f;
        if (detail <= 0.0f)
            detail = 1.0f;
        if (smoothness <= 0.0f)
            smoothness = 1.0f;

        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        voronoiKernel<<<gridSize, blockSize>>>(d_data, m_Width, m_Height, scale, detail, roughness, lacunarity, smoothness);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    Texture &Texture::mix(const Texture &value, const Texture &source)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        mixKernel<<<gridSize, blockSize>>>(d_data, value.d_data, source.d_data,
                                           m_Width, m_Height, value.m_Width, value.m_Height,
                                           source.m_Width, source.m_Height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    Texture &Texture::grayscale()
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        grayscaleKernel<<<gridSize, blockSize>>>(d_data, m_Width, m_Height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    Texture &Texture::blur(float radius)
    {
        if (radius <= 0.0f)
            return *this;

        int kernelSize = int(std::ceil(radius) * 2 + 1);
        std::vector<float> kernel(kernelSize);
        float sigma = radius / 2.0f;
        float sum = 0.0f;

        int half = kernelSize / 2;
        for (int i = 0; i < kernelSize; ++i)
        {
            float x = float(i - half);
            kernel[i] = std::exp(-0.5f * (x * x) / (sigma * sigma));
            sum += kernel[i];
        }
        for (float &k : kernel)
            k /= sum;

        float *d_kernel;
        float *d_temp;
        size_t kernelBytes = kernelSize * sizeof(float);
        size_t dataBytes = m_Width * m_Height * PTEX_TEXTURE_CHANNELS * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_kernel, kernelBytes));
        CUDA_CHECK(cudaMalloc(&d_temp, dataBytes));
        CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice));

        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        // Horizontal pass
        blurHorizontalKernel<<<gridSize, blockSize>>>(d_temp, d_data, d_kernel, m_Width, m_Height, kernelSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Vertical pass
        blurVerticalKernel<<<gridSize, blockSize>>>(d_data, d_temp, d_kernel, m_Width, m_Height, kernelSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_kernel);
        cudaFree(d_temp);
        return *this;
    }

    float *Texture::end()
    {
        if (!d_data)
        {
            throw std::runtime_error("Device data is null");
        }

        size_t size = m_Width * m_Height * PTEX_TEXTURE_CHANNELS;
        size_t bytes = size * sizeof(float);

        CUDA_CHECK(cudaMemcpy(m_Data, d_data, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        return m_Data;
    }

    const float *Texture::getData() const
    {
        return m_Data;
    }
}