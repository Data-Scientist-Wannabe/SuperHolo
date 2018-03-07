#pragma once

#include <cstdint>

#define BITMAP_ID 0x4d42

/* see https://en.wikipedia.org/wiki/BMP_file_format */

#pragma pack(push, 1)  
struct bmp_header
{
    uint16_t    id;                 /* identifiy bitmap and dib */
    uint32_t    file_size;          /* size of the bmp file in bytes */
    uint16_t    reserved_1;         /* application specific */
    uint16_t    reserved_2;         /* application specific */
    uint32_t    offset;             /* offset of where the image data can be found */
}; // __attribute__ ((packed));
#pragma pack(pop)  

#pragma pack(push, 1)  
struct dib_header
{
    uint32_t    dib_size;           /* size of this dib header */
    uint32_t    width;              /* bitmap width in pixels */
    uint32_t    height;             /* bitmap height in pixels */
    uint16_t    planes;             /* number of color planes, must be 1 */
    uint16_t    bpp;                /* number of bits per pixel */
    uint32_t    compression;        /* compression method being used */
    uint32_t    image_size;         /* raw image size in bytes */
    int32_t     h_dpm;              /* horizontal resolution (pixels per meter) */
    int32_t     v_dpm;              /* vertical resolution (pixels per meter) */
    uint32_t    pallete_count;      /* number of colors in pallete, or zero for 2^n */
    uint32_t    important_colors;   /* 0 by default, usually ignored */
}; //__attribute__ ((packed));
#pragma pack(pop)  

struct pixel32bpp
{
    union{
        uint32_t raw;
        struct{
            uint8_t b, g, r, a;
        };
    };
};

