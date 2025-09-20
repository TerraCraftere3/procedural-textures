#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <cmath>
#include <cassert>
#include <initializer_list>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace PTex
{
    template <typename T, int Size>
    struct Vector
    {
        union
        {
            T data[Size];

            struct
            {
                T x, y, z, w;
            };
            struct
            {
                T r, g, b, a;
            };
        };

        // Default constructor: zero-initialize
        __host__ __device__ Vector()
        {
            for (int i = 0; i < Size; ++i)
                data[i] = T(0);
        }

        // Initializer list
        __host__ __device__ Vector(std::initializer_list<T> list)
        {
            assert(list.size() <= Size);
            int i = 0;
            for (T v : list)
                data[i++] = v;
        }

        // Single value
        __host__ __device__ Vector(T value)
        {
            for (int i = 0; i < Size; ++i)
                data[i] = value;
        }

        // vec2, vec3, vec4 constructors
        template <int S = Size, typename = std::enable_if_t<S == 2>>
        __host__ __device__ Vector(T x_, T y_) : x(x_), y(y_) {}
        template <int S = Size, typename = std::enable_if_t<S == 3>>
        __host__ __device__ Vector(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
        template <int S = Size, typename = std::enable_if_t<S == 4>>
        __host__ __device__ Vector(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}

        // Mixed constructors
        template <int S = Size, typename = std::enable_if_t<S == 3>>
        __host__ __device__ Vector(const Vector<T, 2> &v, T z_) : x(v.x), y(v.y), z(z_) {}
        template <int S = Size, typename = std::enable_if_t<S == 4>>
        __host__ __device__ Vector(const Vector<T, 2> &v, T z_, T w_) : x(v.x), y(v.y), z(z_), w(w_) {}
        template <int S = Size, typename = std::enable_if_t<S == 4>>
        __host__ __device__ Vector(const Vector<T, 3> &v, T w_) : x(v.x), y(v.y), z(v.z), w(w_) {}

        // Element access
        __host__ __device__ T &operator[](int i) { return data[i]; }
        __host__ __device__ const T &operator[](int i) const { return data[i]; }

        // Arithmetic operators
        __host__ __device__ Vector operator+(const Vector &rhs) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] + rhs[i];
            return r;
        }
        __host__ __device__ Vector operator-(const Vector &rhs) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] - rhs[i];
            return r;
        }
        __host__ __device__ Vector operator*(T s) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] * s;
            return r;
        }
        __host__ __device__ Vector operator*(const Vector &rhs) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] * rhs[i];
            return r;
        }
        __host__ __device__ Vector operator/(T s) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] / s;
            return r;
        }
        __host__ __device__ Vector operator-() const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = -data[i];
            return r;
        }

        // Compound assignment operators
        __host__ __device__ Vector &operator+=(const Vector &rhs)
        {
            for (int i = 0; i < Size; i++)
                data[i] += rhs[i];
            return *this;
        }
        __host__ __device__ Vector &operator-=(const Vector &rhs)
        {
            for (int i = 0; i < Size; i++)
                data[i] -= rhs[i];
            return *this;
        }
        __host__ __device__ Vector &operator*=(const Vector &rhs)
        {
            for (int i = 0; i < Size; i++)
                data[i] *= rhs[i];
            return *this;
        }
        __host__ __device__ Vector &operator/=(const Vector &rhs)
        {
            for (int i = 0; i < Size; i++)
                data[i] /= rhs[i];
            return *this;
        }

        __host__ __device__ Vector &operator+=(T s)
        {
            for (int i = 0; i < Size; i++)
                data[i] += s;
            return *this;
        }
        __host__ __device__ Vector &operator-=(T s)
        {
            for (int i = 0; i < Size; i++)
                data[i] -= s;
            return *this;
        }
        __host__ __device__ Vector &operator*=(T s)
        {
            for (int i = 0; i < Size; i++)
                data[i] *= s;
            return *this;
        }
        __host__ __device__ Vector &operator/=(T s)
        {
            for (int i = 0; i < Size; i++)
                data[i] /= s;
            return *this;
        }

        __host__ __device__ T length() const
        {
            T t = 0;
            for (int i = 0; i < Size; i++)
                t += data[i] * data[i];
            return sqrt(t);
        }

        __host__ __device__ Vector<T, Size> lerp(const Vector<T, Size> &other, T alpha) const
        {
            Vector<T, Size> result;
            for (int i = 0; i < Size; i++)
                result.data[i] = data[i] + alpha * (other.data[i] - data[i]);
            return result;
        }
    };

    // Scalar * vector
    template <typename T, int Size>
    __host__ __device__ Vector<T, Size> operator*(T s, const Vector<T, Size> &v)
    {
        Vector<T, Size> r;
        for (int i = 0; i < Size; i++)
            r[i] = s * v[i];
        return r;
    }

    template <typename T, int Size>
    __host__ __device__
        Vector<T, Size>
        operator/(const Vector<T, Size> &a, const Vector<T, Size> &b)
    {
        Vector<T, Size> result;
        for (int i = 0; i < Size; ++i)
        {
            result[i] = a[i] / b[i];
        }
        return result;
    }

    using vec2 = Vector<float, 2>;
    using vec3 = Vector<float, 3>;
    using vec4 = Vector<float, 4>;
    using vec2d = Vector<double, 2>;
    using vec3d = Vector<double, 3>;
    using vec4d = Vector<double, 4>;
    using vec2i = Vector<int, 2>;
    using vec3i = Vector<int, 3>;
    using vec4i = Vector<int, 4>;
}

#endif