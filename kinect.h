#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include "libfreenect.h"

#define KINECT_WIDTH  640
#define KINECT_HEIGHT 480

struct Kinect
{
    pthread_mutex_t lock;

    freenect_context *fn_ctx;

    freenect_device *fn_dev;
    
    bool running;

    freenect_led_options led;

    uint32_t depth_timestamp;

    uint32_t video_timestamp;

    uint16_t depth_buffer[KINECT_WIDTH][KINECT_HEIGHT];

    uint8_t  video_buffer[3 * KINECT_WIDTH * KINECT_HEIGHT];    
};

extern void * kinect_threadfunc(void * arg);

extern int kinect_init(struct Kinect * kinect, freenect_context *, int index);

extern void kinect_destroy(struct Kinect * kinect );