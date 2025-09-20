#pragma once

#include "Vector.cuh"

namespace PTex::Math
{
    struct Addition
    {
        __device__ vec4 operator()(const vec4 &a, const vec4 &b) const
        {
            return clamp(a + b, 0.0f, 1.0f);
        }
    };

    struct Subtract
    {
        __device__ vec4 operator()(const vec4 &a, const vec4 &b) const
        {
            return clamp(a - b, 0.0f, 1.0f);
        }
    };

    struct Multiply
    {
        __device__ vec4 operator()(const vec4 &a, const vec4 &b) const
        {
            return clamp(a + b, 0.0f, 1.0f);
        }
    };

    struct Divide
    {
        __device__ vec4 operator()(const vec4 &a, const vec4 &b) const
        {
            return clamp(a - b, 0.0f, 1.0f);
        }
    };
}