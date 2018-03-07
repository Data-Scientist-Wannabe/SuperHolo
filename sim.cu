#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "pattern.h"

#define CUDART_2PI      6.2831853071795865e+0
#define CUDART_PI       3.1415926535897931e+0
#define CUDART_PIO6     0.52359877559829887e+0

#define ARRAY_WIDTH     684
#define ARRAY_HEIGHT    608
#define ARRAY_SIZE      (ARRAY_WIDTH * ARRAY_HEIGHT)

#define POINT_COUNT     1

#define ARRAY_WIDTH_INV     0.0014619883040935673
#define ARRAY_HEIGHT_INV    0.0016447368421052632

#define PATTERN_WIDTH   0.0065718
#define PATTERN_HEIGHT  0.003699 

#define N_BLOCKS    256     //
#define N_THREADS   512     // Threads per block

#define LAMBDA          632.8e-9
#define LAMBDA_INV      1.580278128950695e6
#define REF_BEAM_ANGLE  CUDART_PIO6

#define PLANE_CONST (2.0 * CUDART_PI / LAMBDA * sin(REF_BEAM_ANGLE))
#define VAL_CONST   (CUDART_2PI / LAMBDA)

__device__ double plane(double x)
{
    return cos(PLANE_CONST * x);
}

__device__ double distance(double2 uv, double3 point)
{
    double x = uv.x - point.x;
    double y = uv.y - point.y;
    return sqrt(x*x + y*y + point.z * point.z);
}

__device__ double val(double2 uv, double3 point)
{
    double d = distance(uv, point);
    return sin(( d - LAMBDA * floor(d * LAMBDA_INV)) * VAL_CONST);
}

__device__ double intensity(double2 uv, double3 point)
{
    double d = distance(uv, point);
    return 1.0 / (d * d);
}

__global__ void simulation(double * pattern, double4 * points, int count)
{
    for(int x = threadIdx.x; x < ARRAY_WIDTH; x += blockDim.x)
    {
        for(int y = blockIdx.x; y < ARRAY_HEIGHT; y+= gridDim.x)
        {
            double2 uv = {x * ARRAY_WIDTH_INV * PATTERN_WIDTH, y * ARRAY_HEIGHT_INV * PATTERN_HEIGHT};
            int index = y * ARRAY_WIDTH + x;

            pattern[index]= plane(uv.x);

            for(int i = 0; i < count; i++)
            {
                double3 point = make_double3(points[i].x, points[i].y, points[i].z);
                pattern[index] += points[i].w * val(uv, point) * intensity(uv, point);
            }
        }
    }

    return;
}

int main(void)
{
    double * d_pattern;

    double4 * h_points;
    double4 * d_points;

    dim3 dimGrid(N_BLOCKS);
    dim3 dimBlock(N_THREADS);

    pattern pat(ARRAY_WIDTH, ARRAY_HEIGHT);

    h_points  =  (double4*)malloc(POINT_COUNT * sizeof(double4));

    cudaMalloc((void**)&d_pattern,  ARRAY_SIZE * sizeof(double));
    cudaMalloc((void**)&d_points,   POINT_COUNT* sizeof(double4));

    h_points[0] = make_double4(PATTERN_WIDTH / 2.0, PATTERN_HEIGHT / 2.0, 0.002, 1.0);

    cudaMemcpy(d_points, h_points, POINT_COUNT * sizeof(double4), cudaMemcpyHostToDevice);

    // call method with dumb syntax
    simulation<<<dimGrid, dimBlock>>>(d_pattern, d_points, POINT_COUNT);

    cudaDeviceSynchronize();

    cudaMemcpy(pat.data, d_pattern, ARRAY_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    free(h_points);
    cudaFree(d_pattern);
    cudaFree(d_points);

    pat.export_bmp("sim.bmp");
    pat.save("sim.out");

    return 0;
}
