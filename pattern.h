#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#define MAGIC 0x5ADB01

struct pattern_header
{
    uint32_t magic;
    uint32_t width;
    uint32_t height;
    uint32_t _pad_0;
};

class pattern{
public:
    uint32_t    width;
    uint32_t    height;

    float *    data;
    uint32_t    size;

public:
    pattern(void) : width(0), height(0), data(nullptr), size(0) {};
    pattern(uint32_t const width, uint32_t const height);
    ~pattern(void);

    int load(const char * const pFilename);
    int save(const char * const pFilename);
    int export_bmp(const char * const pFilename);
};