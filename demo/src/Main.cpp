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
    try
    {
        Texture tex = Texture(512, 512).noise();
        writePPM("output.ppm", tex);
        printf("Wrote file \"output.ppm\"...\n");
    }
    catch (std::runtime_error err)
    {
        printf("Error occured!\n");
        throw err;
    }
    return 0;
}