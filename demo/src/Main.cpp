#include <iostream>
#include <ptex/PTex.h>

#include <fstream>
#include <algorithm>

void writePPM(const std::string &filename, const PTex::Texture &texture)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open file");

    file << "P6\n"
         << texture.width() << " " << texture.height() << "\n255\n";

    auto data = texture.data();
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
    Texture tex = Texture().gradient(45.0f);
    writePPM("output.ppm", tex);
    return 0;
}