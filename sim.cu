#include "sim.h"
#include "interop.h"
#include "kernel.h"
#include <cstdlib>

/* Kinect transformation from http://nicolas.burrus.name/index.php/Research/KinectCalibration */
#define FX_RGB 5.2921508098293293e+02
#define FY_RGB 5.2556393630057437e+02
#define CX_RGB 3.2894272028759258e+02
#define CY_RGB 2.6748068171871557e+02
#define K1_RGB 2.6451622333009589e-01
#define K2_RGB -8.3990749424620825e-01
#define P1_RGB -1.9922302173693159e-03
#define P2_RGB 1.4371995932897616e-03
#define K3_RGB 9.1192465078713847e-01

#define FX_D 5.9421434211923247e+02
#define FY_D 5.9104053696870778e+02
#define CX_D 3.3930780975300314e+02
#define CY_D 2.4273913761751615e+02
#define K1_D -2.6386489753128833e-01
#define K2_D 9.9966832163729757e-01
#define P1_D -7.6275862143610667e-04
#define P2_D 5.0350940090814270e-03
#define K3_D -1.3053628089976321e+00

float4 calcPoint(uint16_t depth, int x, int y)
{
	float4 point;
	point.x = (x - CX_D) * depth / FX_D;
	point.y = (y - CY_D) * depth / FY_D;
	point.z = depth;
	point.w = 1.0f;

	return point;
}

Simulation::Simulation(int width, int height, int max_points)
: width(width), height(height)
{
    createGLTextureForCUDA(&this->gl_tex, &this->cuda_tex_resource, width, height);
    cudaMalloc(&this->cuda_dev_render_buffer, this->width * this->height * sizeof(float));

	cudaMalloc((void**)&d_points, max_points * sizeof(float4));
	h_points = (float4*)malloc(max_points * sizeof(float4));
}

void Simulation::generateImage(int point_count)
{
	if(point_count > this->point_count)
		point_count = this->point_count;

    launch_kernel(N_BLOCKS, N_THREADS, (float*)cuda_dev_render_buffer, d_points, point_count);
	
	cudaArray * texture_ptr;
	CUDA_CALL(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
	CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

	int size_tex_data = width * height * sizeof(float);

	CUDA_CALL(cudaMemcpyToArray(
		texture_ptr, 0, 0, 
		cuda_dev_render_buffer, 
		size_tex_data, 
		cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
}

void Simulation::setPoints(Kinect * kinect, uint16_t max)
{
	/* int i = 0;

	pthread_mutex_lock(&kinect->lock);

	for(int x = 0; x < KINECT_WIDTH; x++)
	{
		for(int y = 0; y < KINECT_HEIGHT; y++)
		{
			uint16_t depth = kinect->depth_buffer[x][y];
			if(depth < max) {
				h_points[i++] = calcPoint(depth, x, y);
				h_points[i - 1].z = (x + y) * 10.0f;
			}
		}
	}

	pthread_mutex_unlock(&kinect->lock);

	this->point_count = i; */

	h_points[0] = make_float4(PATTERN_WIDTH * 0.33f, PATTERN_HEIGHT * 0.20f, 0.30f, 0.33f);
    h_points[1] = make_float4(PATTERN_WIDTH * 0.50f, PATTERN_HEIGHT * 0.40f, 0.30f, 0.33f);
    h_points[2] = make_float4(PATTERN_WIDTH * 0.66f, PATTERN_HEIGHT * 0.60f, 0.30f, 0.33f);
	h_points[3] = make_float4(PATTERN_WIDTH * 0.50f, PATTERN_HEIGHT * 0.80f, 0.30f, 0.33f);

	this->point_count = 4;

	cudaMemcpy(d_points, h_points, this->point_count * sizeof(float4), cudaMemcpyHostToDevice);
}