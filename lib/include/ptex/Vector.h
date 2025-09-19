/*

taken from TerraCraftere3/Pathway (https://github.com/TerraCraftere3/Pathway/blob/main/Pathway/src/Math/Vector.cuh)

*/

#ifndef VECTOR_H
#define VECTOR_H
#include <cmath>

#include <cassert>

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
        Vector()
        {
            for (int i = 0; i < Size; ++i)
                data[i] = T(0);
        }

        // Initializer list
        Vector(std::initializer_list<T> list)
        {
            assert(list.size() <= Size);
            int i = 0;
            for (T v : list)
                data[i++] = v;
        }

        // Single value
        Vector(T value)
        {
            for (int i = 0; i < Size; ++i)
                data[i] = value;
        }

        // vec2, vec3, vec4 constructors
        template <int S = Size, typename = std::enable_if_t<S == 2>>
        Vector(T x_, T y_) : x(x_), y(y_) {}
        template <int S = Size, typename = std::enable_if_t<S == 3>>
        Vector(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
        template <int S = Size, typename = std::enable_if_t<S == 4>>
        Vector(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}

        // Mixed constructors
        template <int S = Size, typename = std::enable_if_t<S == 3>>
        Vector(const Vector<T, 2> &v, T z_) : x(v.x), y(v.y), z(z_) {}
        template <int S = Size, typename = std::enable_if_t<S == 4>>
        Vector(const Vector<T, 2> &v, T z_, T w_) : x(v.x), y(v.y), z(z_), w(w_) {}
        template <int S = Size, typename = std::enable_if_t<S == 4>>
        Vector(const Vector<T, 3> &v, T w_) : x(v.x), y(v.y), z(v.z), w(w_) {}

        // Element access
        T &operator[](int i) { return data[i]; }
        const T &operator[](int i) const { return data[i]; }

        // Arithmetic operators
        Vector operator+(const Vector &rhs) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] + rhs[i];
            return r;
        }
        Vector operator-(const Vector &rhs) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] - rhs[i];
            return r;
        }
        Vector operator*(T s) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] * s;
            return r;
        }
        Vector operator*(const Vector &rhs) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] * rhs[i];
            return r;
        }
        Vector operator/(T s) const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = data[i] / s;
            return r;
        }
        Vector operator-() const
        {
            Vector r;
            for (int i = 0; i < Size; i++)
                r[i] = -data[i];
            return r;
        }

        // Compound assignment operators
        Vector &operator+=(const Vector &rhs)
        {
            for (int i = 0; i < Size; i++)
                data[i] += rhs[i];
            return *this;
        }
        Vector &operator-=(const Vector &rhs)
        {
            for (int i = 0; i < Size; i++)
                data[i] -= rhs[i];
            return *this;
        }
        Vector &operator*=(const Vector &rhs)
        {
            for (int i = 0; i < Size; i++)
                data[i] *= rhs[i];
            return *this;
        }
        Vector &operator/=(const Vector &rhs)
        {
            for (int i = 0; i < Size; i++)
                data[i] /= rhs[i];
            return *this;
        }

        Vector &operator+=(T s)
        {
            for (int i = 0; i < Size; i++)
                data[i] += s;
            return *this;
        }
        Vector &operator-=(T s)
        {
            for (int i = 0; i < Size; i++)
                data[i] -= s;
            return *this;
        }
        Vector &operator*=(T s)
        {
            for (int i = 0; i < Size; i++)
                data[i] *= s;
            return *this;
        }
        Vector &operator/=(T s)
        {
            for (int i = 0; i < Size; i++)
                data[i] /= s;
            return *this;
        }
    };

    // Scalar * vector
    template <typename T, int Size>
    Vector<T, Size> operator*(T s, const Vector<T, Size> &v)
    {
        Vector<T, Size> r;
        for (int i = 0; i < Size; i++)
            r[i] = s * v[i];
        return r;
    }

    template <typename T, int Size>
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