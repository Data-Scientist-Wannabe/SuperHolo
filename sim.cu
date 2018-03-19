#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "pattern.h"

#define POINT_COUNT     3

#define CUDART_2PI      6.2831853071795865e+0
#define CUDART_PI       3.1415926535897931e+0
#define CUDART_PIO6     0.52359877559829887e+0

#define ARRAY_WIDTH     (684)
#define ARRAY_HEIGHT    (608)
#define ARRAY_SIZE      (ARRAY_WIDTH * ARRAY_HEIGHT)

#define ARRAY_WIDTH_INV     (1.0 / ARRAY_WIDTH)  //0.0014619883040935673
#define ARRAY_HEIGHT_INV    (1.0 / ARRAY_HEIGHT) //0.0016447368421052632

#define PATTERN_WIDTH   6571.8e-6
#define PATTERN_HEIGHT  3699e-6
#define N_BLOCKS    256     //
#define N_THREADS   512     // Threads per block

#define LAMBDA          632.8e-9
#define LAMBDA_INV      (1.0 / LAMBDA) //1.580278128950695e6
#define TWO_LAMBDA_INV  (2.0 / LAMBDA) //3.160556257901391e6
#define REF_BEAM_ANGLE  (0.00565003)    // ~0.33 degrees

#define PLANE_CONST (TWO_LAMBDA_INV * sin(REF_BEAM_ANGLE))
#define VAL_CONST   (TWO_LAMBDA_INV)

__device__ double plane(double x)
{
    return cospi(PLANE_CONST * x);
}

__device__ double distance(double2 uv, double3 point)
{
    return norm3d(uv.x - point.x, uv.y - point.y, point.z);
}

__device__ double val(double2 uv, double3 point)
{
    double d = distance(uv, point);
    return sinpi(remainder(d, LAMBDA) * VAL_CONST);
}

__device__ double intensity(double2 uv, double3 point)
{
    double x = uv.x - point.x;
    double y = uv.y - point.y;
    return 1.0 / (x * x + y * y + point.z * point.z);
}

__global__ void simulation(double * pattern, double4 * points, int count)
{
    for(int x = threadIdx.x; x < ARRAY_WIDTH; x += blockDim.x)
    {
        for(int y = blockIdx.x; y < ARRAY_HEIGHT; y+= gridDim.x)
        {
            double2 uv = make_double2(x * ARRAY_WIDTH_INV * PATTERN_WIDTH, y * ARRAY_HEIGHT_INV * PATTERN_HEIGHT);
            int index = y * ARRAY_WIDTH + x;

            pattern[index]= plane(uv.x);

            for(int i = 0; i < count; i++)
            {
                double3 point = make_double3(points[i].x, points[i].y, points[i].z);
                pattern[index] += points[i].w * intensity(uv, point) * val(uv, point);
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

    h_points[0] = make_double4(PATTERN_WIDTH * 0.50, PATTERN_HEIGHT * 0.25, 0.30, 0.33);
    h_points[1] = make_double4(PATTERN_WIDTH * 0.50, PATTERN_HEIGHT * 0.50, 0.30, 0.33);
    h_points[2] = make_double4(PATTERN_WIDTH * 0.50, PATTERN_HEIGHT * 0.75, 0.30, 0.33);

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
