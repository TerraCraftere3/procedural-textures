#pragma once

#include <vector>

#include "Vector.cuh"
#include "Core.h"

#define PTEX_TEXTURE_CHANNELS 4

#ifdef PTEX_USE_OPENGL
#include "glad/glad.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

namespace PTex
{

    class PTEX_API Texture
    {
    public:
        Texture(int width = 512, int height = 512);
        ~Texture();
        Texture(const Texture &);
        Texture &operator=(const Texture &other);

        int width() const { return m_Width; }
        int height() const { return m_Height; }

        // CPU data access (only valid after calling end())
        const float *getData() const;
        int getTextureID() const;

        // GPU to CPU data transfer
        float *copy();
        int end();

        Texture &setData(const float *data, int size);
        Texture &fill(vec4 color);
        Texture &zero();
        Texture &gradient(vec4 colA, vec4 colB, float angle = 0.0f);
        Texture &noise(float scale = 5.0f, float detail = 2.0f, float roughness = 0.5f, float lacunarity = 2.0f, float distortion = 0.0f);
        Texture &voronoi(float scale = 5.0f, float detail = 1.0f, float roughness = 0.5f, float lacunarity = 2.0f, float smoothness = 1.0f);
        Texture &mix(const Texture &value, const Texture &source);
        Texture &add(const Texture &other);
        Texture &sub(const Texture &other);
        Texture &multi(const Texture &other);
        Texture &divide(const Texture &other);
        Texture &grayscale();
        Texture &blur(float radius = 1.0f);

    private:
        int m_Width, m_Height;
        float *m_Data; // CPU data
        float *d_data; // GPU data pointer
#ifdef PTEX_USE_OPENGL
        GLuint m_GLTexture = 0;                         // existing GL texture
        cudaGraphicsResource *m_CUDAResource = nullptr; // CUDA-GL interop handle
#endif
    };
}