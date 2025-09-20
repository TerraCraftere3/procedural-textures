#include "ptex/Texture.cuh"
#include "ptex/Kernel.cuh"

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

namespace PTex
{

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
            cudaFree(d_data);
        if (m_Data)
            cudaFreeHost(m_Data);
#ifdef PTEX_USE_OPENGL
        if (m_CUDAResource)
            cudaGraphicsUnregisterResource(m_CUDAResource);
        if (m_GLTexture)
            glDeleteTextures(1, &m_GLTexture);
#endif
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

    Texture &Texture::add(const Texture &other)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        mathFunctionKernel<<<gridSize, blockSize>>>(d_data, other.d_data, m_Width, m_Height, other.m_Width, other.m_Height, Math::Addition{});
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    Texture &Texture::sub(const Texture &other)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        mathFunctionKernel<<<gridSize, blockSize>>>(d_data, other.d_data, m_Width, m_Height, other.m_Width, other.m_Height, Math::Subtract{});
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    Texture &Texture::multi(const Texture &other)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        mathFunctionKernel<<<gridSize, blockSize>>>(d_data, other.d_data, m_Width, m_Height, other.m_Width, other.m_Height, Math::Multiply{});
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *this;
    }

    Texture &Texture::divide(const Texture &other)
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((m_Width + blockSize.x - 1) / blockSize.x,
                      (m_Height + blockSize.y - 1) / blockSize.y);

        mathFunctionKernel<<<gridSize, blockSize>>>(d_data, other.d_data, m_Width, m_Height, other.m_Width, other.m_Height, Math::Divide{});
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

    float *Texture::copy()
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

    int Texture::end()
    {
#ifdef PTEX_USE_OPENGL
        if (m_GLTexture == 0)
        {
            glGenTextures(1, &m_GLTexture);
            glBindTexture(GL_TEXTURE_2D, m_GLTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            // Allocate GPU storage
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, nullptr);

            // Register with CUDA
            CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_CUDAResource, m_GLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
        }

        // Map CUDA resource
        CUDA_CHECK(cudaGraphicsMapResources(1, &m_CUDAResource, 0));
        cudaArray_t array;
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, m_CUDAResource, 0, 0));

        // Copy your device data into the GL texture
        CUDA_CHECK(cudaMemcpy2DToArray(
            array,
            0, 0,
            d_data,
            m_Width * PTEX_TEXTURE_CHANNELS * sizeof(float),
            m_Width * PTEX_TEXTURE_CHANNELS * sizeof(float),
            m_Height,
            cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_CUDAResource, 0));

        return m_GLTexture;
#else
#pragma message("No OpenGL Support for Texture.cu!!!");
        return 0;
#endif
    }

    const float *Texture::getData() const
    {
        return m_Data;
    }

    int Texture::getTextureID() const
    {
        return m_GLTexture;
    }

}