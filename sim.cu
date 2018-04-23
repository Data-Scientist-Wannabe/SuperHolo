#include "sim.h"
#include <cstdlib>
#include "interop.h"
#include "kernel.h"

Simulation::Simulation(int width, int height, int max_points)
: width(width), height(height), max_point_count(max_points), point_count(0)
{
    createGLTextureForCUDA(&gl_tex, &cuda_tex_resource, width, height);

    cudaMalloc((void**)&cuda_dev_render_buffer, width * height * sizeof(float));

	cudaMalloc((void**)&d_points, max_points * sizeof(float4) * 2);
	cudaMalloc((void**)&d_transform, sizeof(float4) * 4);

	amp = 1.0f;
}

Simulation::~Simulation()
{
	cudaFree(this->d_points);
	cudaFree(this->d_pattern);
	cudaFree(this->d_transform);
}

void Simulation::generateImage(void)
{
	launch_transform(64, 64, 
		(float4*)d_points, 
		(float4*)d_points + max_point_count,
		(float4*)d_transform, 
		amp,
		point_count);

    launch_kernel(N_BLOCKS, N_THREADS,
		(float*)cuda_dev_render_buffer,
		(float4*)d_points + max_point_count,
		point_count);
	
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

void Simulation::setPoints(void * points, int count)
{
	if(count > this->max_point_count)
		count = this->max_point_count;

	this->point_count = count;

	cudaMemcpy(d_points, points, count * sizeof(float4), cudaMemcpyHostToDevice);
}

void Simulation::SetTransformation(Ruined::Math::Matrix mat)
{
	cudaMemcpy(d_transform, mat.m, sizeof(float4) * 4, cudaMemcpyHostToDevice);
}