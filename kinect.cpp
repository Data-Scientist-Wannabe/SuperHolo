#include "kinect.h"
#include <string.h>
#include <stdio.h>

void depth_cb(freenect_device*, void*, uint32_t);

void video_cb(freenect_device*, void*, uint32_t);

void * kinect_threadfunc(void * arg)
{
	struct Kinect * kinect = (struct Kinect*)arg;

	kinect->running = true;
	kinect->led = LED_GREEN;

	freenect_set_depth_callback(kinect->fn_dev, depth_cb);
	freenect_set_video_callback(kinect->fn_dev, video_cb);
	freenect_set_depth_mode(kinect->fn_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM));
	freenect_set_video_mode(kinect->fn_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));

	freenect_start_depth(kinect->fn_dev);
	freenect_start_video(kinect->fn_dev);

	while(kinect->running && freenect_process_events(kinect->fn_ctx) >= 0)
	{
	}

	freenect_set_led(kinect->fn_dev, LED_RED);
	freenect_stop_depth(kinect->fn_dev);
	freenect_stop_video(kinect->fn_dev);

	kinect->running = false;

	return nullptr;
}

#define KINECT_DEPTH_BUFFER_SIZE	(640 * 480 * sizeof(uint16_t))

void depth_cb(freenect_device* dev, void* data, uint32_t timestamp)
{
	struct Kinect * kinect = (struct Kinect * )freenect_get_user(dev);

	pthread_mutex_lock(&kinect->lock);

	memcpy(kinect->depth_buffer, data, sizeof(kinect->depth_buffer));

	kinect->depth_timestamp = timestamp;

	pthread_mutex_unlock(&kinect->lock);
}

void video_cb(freenect_device* dev, void* data, uint32_t timestamp)
{
	//struct Kinect * kinect = (struct Kinect * )freenect_get_user(dev);

	// pthread_mutex_lock(&kinect->lock);

	// memcpy(kinect->video_buffer, data, sizeof(kinect->video_buffer));

	// kinect->video_timestamp = timestamp;

	// printf("color recieved\n");

	// pthread_mutex_unlock(&kinect->lock);
}

extern int kinect_init(struct Kinect * kinect, freenect_context * fn_ctx, int index )
{
	int ret;

	kinect->fn_ctx = fn_ctx;

	if(ret = pthread_mutex_init(&kinect->lock, NULL))
		return ret;

	if(ret = freenect_open_device(kinect->fn_ctx, &kinect->fn_dev, index))
		return ret;

	freenect_set_user(kinect->fn_dev, kinect);

	return 0;
}

extern void kinect_destroy(struct Kinect * kinect)
{
	pthread_mutex_destroy(&kinect->lock);
}