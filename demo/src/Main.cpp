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
    fflush(stdout);
    auto start = std::chrono::high_resolution_clock::now();

    Texture tex(512, 512);
    tex.gradient(vec4(1.0f, 0.3f, 0.2f, 1.0f), vec4(0.2f, 0.3f, 1.0f, 1.0f), 45.0f)
        .mix(Texture(256, 256).voronoi(), Texture(256, 256).noise())
        .blur(25.0f)
        .end();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    writePPM("output.ppm", tex);
    printf("Wrote file \"output.ppm\"...\n");
    printf("Texture creation took %.3f ms\n", elapsed.count());
    return 0;
}