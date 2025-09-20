#include <ptex/PTex.h>

#include <fstream>
#include <algorithm>
#include <chrono>
#include <iostream>

void writePPM(const std::string &filename, const PTex::Texture &texture)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open file");

    file << "P6\n"
         << texture.width() << " " << texture.height() << "\n255\n";

    auto data = texture.getData();
    for (int i = 0; i < texture.width() * texture.height(); i++)
    {
        unsigned char r = static_cast<unsigned char>(std::clamp(data[i * PTEX_TEXTURE_CHANNELS + 0], 0.0f, 1.0f) * 255.0f);
        unsigned char g = static_cast<unsigned char>(std::clamp(data[i * PTEX_TEXTURE_CHANNELS + 1], 0.0f, 1.0f) * 255.0f);
        unsigned char b = static_cast<unsigned char>(std::clamp(data[i * PTEX_TEXTURE_CHANNELS + 2], 0.0f, 1.0f) * 255.0f);

        file.put(r);
        file.put(g);
        file.put(b);
    }
}

int main()
{
    using namespace PTex;

    printf("Starting Texture Creation\n");
    auto start = std::chrono::high_resolution_clock::now();

    int size = 512;
#define SIZED_TEXTURE() Texture(size, size)
    Texture tex(size, size);
    tex.gradient(vec4(1.0f, 0.3f, 0.2f, 1.0f), vec4(0.2f, 0.3f, 1.0f, 1.0f), 45.0f)
        .mix(SIZED_TEXTURE().voronoi(), SIZED_TEXTURE().noise())
        .blur(25.0f)
        .multi(SIZED_TEXTURE().voronoi())
        .end();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Texture creation took %.3f ms\n", elapsed.count());
    writePPM("output.ppm", tex);
    printf("Wrote file \"output.ppm\"...\n");
    return 0;
}