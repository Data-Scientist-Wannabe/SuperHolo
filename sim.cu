#include "sim.h"
#include "interop.h"
#include "kernel.h"
#include <cstdlib>

Simulation::Simulation(int width, int height, int max_points)
: width(width), height(height), max_point_count(max_points)
{
    createGLTextureForCUDA(&this->gl_tex, &this->cuda_tex_resource, width, height);
    cudaMalloc(&this->cuda_dev_render_buffer, this->width * this->height * sizeof(float));

	cudaMalloc((void**)&d_points, max_points * sizeof(float4));
}

void Simulation::generateImage(int point_count)
{
	if(point_count > this->max_point_count)
		point_count = this->max_point_count;

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

extern double GLFW_TIME;

void Simulation::setPoints(float4 * points, int point_count)
{
	if(point_count > this->max_point_count)
		point_count = this->max_point_count;

	cudaMemcpy(d_points, points, point_count * sizeof(float4), cudaMemcpyHostToDevice);
}