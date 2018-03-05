#pragma once

#define FLAG_ERROR      0x80000000‬
#define IS_ERROR(x)     (x & FLAG_ERROR)
#define IS_SUCCESS      (~(x & FLAG_ERROR))

#define ERROR_FILE      0x80000002
#define ERROR_MEMORY    0x80000003
#define ERROR_INPUT     0x80000004
#define ERROR_FORMAT    0x80000005

#define SUCCESS         0x0