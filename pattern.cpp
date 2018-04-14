#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "error.h"
#include "bmp.h"
#include "pattern.h"

int pattern::load(const char * const pFilename)
{
    size_t file_size;
    struct pattern_header header;

    std::ifstream file(pFilename, std::ios::binary  |std::ifstream::ate);
    if(!file.is_open()) {
        return ERROR_FILE;
    }

    file_size = (size_t)file.tellg();
    if(file_size < sizeof(struct pattern_header)){
        file.close();
        return ERROR_FILE;
    }

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(&header), sizeof(struct pattern_header));
    if(MAGIC != header.magic){
        file.close();
        return ERROR_FORMAT;
    }

    this->width = header.width;
    this->height = header.height;
    this->size = header.width * header.height * sizeof(float);

    this->data = reinterpret_cast<float *>(malloc(this->size));

    if(!this->data){
        file.close();
        return ERROR_MEMORY;       
    }

    file.seekg(sizeof(pattern_header), std::ios::beg);
    file.read(reinterpret_cast<char *>(this->data), this->size);
    file.close();

    return SUCCESS;
}

int pattern::save(const char * const pFilename)
{
    struct pattern_header header;
    header.magic = MAGIC;
    header.width = this->width;
    header.height = this->height;
    header._pad_0 = 0;

    std::ofstream file(pFilename, std::ios::binary | std::ios::trunc);
    if(!file.is_open()) {
        return ERROR_FILE;
    }

    file.write(reinterpret_cast<char*>(&header), sizeof(struct pattern_header));

    file.write(reinterpret_cast<char *>(this->data), this->size);

    file.close();

    return SUCCESS;
}

bool amp_sort(float i, float j)
{
    return (fabs(i) < fabs(j));
}

float find_99_7_percentile(float * data, uint32_t count)
{
    std::vector<float> v(data, data + count);
    std::sort(v.begin(), v.end(), amp_sort);
    return fabs(v[254.0f/255.0f * count]);
}

int pattern::export_bmp(const char * const pFilename)
{
    struct bmp_header bmp;
    struct dib_header dib;

    struct pixel32bpp * pixel_data;

    float max_amp = abs(this->data[0]);

    std::ofstream file(pFilename, std::ios::binary | std::ios::trunc);
    if(!file.is_open()) {
        return ERROR_FILE;
    }

    pixel_data = reinterpret_cast<struct pixel32bpp *>
        (malloc(sizeof(struct pixel32bpp) * this->width * this->height));
    if(!pixel_data) {
        file.close();
        return ERROR_MEMORY;
    }

    bmp.id = BITMAP_ID;
    bmp.offset = sizeof(bmp_header) + sizeof(dib_header);
    bmp.file_size = bmp.offset + 4 * this->width * this->height;

    dib.dib_size = sizeof(dib_header);
    dib.width = this->width;
    dib.height = this->height;
    dib.planes = 1;
    dib.bpp = 32;
    dib.compression = 0;
    dib.image_size = 4 * this->width * this->height;
    dib.h_dpm = 2835;
    dib.v_dpm = 2835;
    dib.pallete_count  = 0;
    dib.important_colors = 0;

    /* for(uint32_t c = 1; c < this->width * this->height; c++)
    {
        float val = fabs(this->data[c]);
        if(val > max_amp) {
            max_amp = val;
        }
    } */

    max_amp = find_99_7_percentile(this->data, this->width * this->height);
    printf("max_amp = %f\n", max_amp);

    for(uint32_t x = 0; x < this->width; x++)
    {
        for(uint32_t y = 0; y < this->height; y++)
        {
            uint32_t data_pos = x + y * this->width;
            uint32_t pixel_pos = x + (this->height - y - 1) * this->width;

            float amp = fabs(this->data[data_pos]);

            uint8_t val = amp > max_amp ? 0xff : amp / max_amp * 0xff;
            pixel_data[pixel_pos].raw = 0xffffffff;
            pixel_data[pixel_pos].r = pixel_data[pixel_pos].g = pixel_data[pixel_pos].b = val;
        }
    }

    file.write(reinterpret_cast<char*>(&bmp), sizeof(struct bmp_header));

    file.write(reinterpret_cast<char*>(&dib), sizeof(struct dib_header));

    file.write(reinterpret_cast<char *>(pixel_data),
        sizeof(struct pixel32bpp) * this->width * this->height);
    
    file.close();

    free(pixel_data);

    return SUCCESS;
}

pattern::pattern(uint32_t const width, uint32_t const height)
    :width(width), height(height), size(width * height * sizeof(float))
{
    this->data = reinterpret_cast<float *>(malloc(this->size));
}

pattern::~pattern(void)
{
    if(data){
        free(data);
        data = nullptr;
    }
}