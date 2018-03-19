#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "pattern.h"

#define POINT_COUNT     3

#define CUDART_2PI      6.2831853071795865f
#define CUDART_PI       3.1415926535897931f
#define CUDART_PIO6     0.52359877559829887f

#define ARRAY_WIDTH     (684)
#define ARRAY_HEIGHT    (608)
#define ARRAY_SIZE      (ARRAY_WIDTH * ARRAY_HEIGHT)

#define ARRAY_WIDTH_INV     (1.0f / ARRAY_WIDTH)  //0.0014619883040935673
#define ARRAY_HEIGHT_INV    (1.0f / ARRAY_HEIGHT) //0.0016447368421052632

#define PATTERN_WIDTH   6571.8e-6f
#define PATTERN_HEIGHT  3699e-6f
#define N_BLOCKS    256     //
#define N_THREADS   512     // Threads per block

#define LAMBDA          632.8e-9f
#define LAMBDA_INV      (1.0f / LAMBDA) //1.580278128950695e6
#define TWO_LAMBDA_INV  (2.0f / LAMBDA) //3.160556257901391e6
#define REF_BEAM_ANGLE  (0.00565003f)    // ~0.33 degrees

#define PLANE_CONST (TWO_LAMBDA_INV * sin(REF_BEAM_ANGLE))
#define VAL_CONST   (TWO_LAMBDA_INV)

__device__ float plane(float x)
{
    return cospif(PLANE_CONST * x);
}

__device__ float distance(float2 uv, float3 point)
{
    return norm3df(uv.x - point.x, uv.y - point.y, point.z);
}

__device__ float val(float2 uv, float3 point)
{
    float d = distance(uv, point);
    return sinpif(remainderf(d, LAMBDA) * VAL_CONST);
}

__device__ float intensity(float2 uv, float3 point)
{
    float x = uv.x - point.x;
    float y = uv.y - point.y;
    return 1.0f / (x * x + y * y + point.z * point.z);
}

__global__ void simulation(float * pattern, float4 * points, int count)
{
    for(int x = threadIdx.x; x < ARRAY_WIDTH; x += blockDim.x)
    {
        for(int y = blockIdx.x; y < ARRAY_HEIGHT; y+= gridDim.x)
        {
            float2 uv = make_float2(x * ARRAY_WIDTH_INV * PATTERN_WIDTH, y * ARRAY_HEIGHT_INV * PATTERN_HEIGHT);
            int index = y * ARRAY_WIDTH + x;

            pattern[index]= plane(uv.x);

            for(int i = 0; i < count; i++)
            {
                float3 point = make_float3(points[i].x, points[i].y, points[i].z);
                pattern[index] += points[i].w * intensity(uv, point) * val(uv, point);
            }
        }
    }

    return;
}

int main(void)
{
    float * d_pattern;

    float4 * h_points;
    float4 * d_points;

    dim3 dimGrid(N_BLOCKS);
    dim3 dimBlock(N_THREADS);

    pattern pat(ARRAY_WIDTH, ARRAY_HEIGHT);

    h_points  =  (float4*)malloc(POINT_COUNT * sizeof(float4));

    cudaMalloc((void**)&d_pattern,  ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&d_points,   POINT_COUNT* sizeof(float4));

    h_points[0] = make_float4(PATTERN_WIDTH * 0.50f, PATTERN_HEIGHT * 0.25f, 0.30f, 0.33f);
    h_points[1] = make_float4(PATTERN_WIDTH * 0.50f, PATTERN_HEIGHT * 0.50f, 0.30f, 0.33f);
    h_points[2] = make_float4(PATTERN_WIDTH * 0.50f, PATTERN_HEIGHT * 0.75f, 0.30f, 0.33f);

    cudaMemcpy(d_points, h_points, POINT_COUNT * sizeof(float4), cudaMemcpyHostToDevice);

    // call method with dumb syntax
    simulation<<<dimGrid, dimBlock>>>(d_pattern, d_points, POINT_COUNT);

    cudaDeviceSynchronize();

    cudaMemcpy(pat.data, d_pattern, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_points);
    cudaFree(d_pattern);
    cudaFree(d_points);

    pat.export_bmp("sim_f32.bmp");
    pat.save("sim_f32.out");

    return 0;
}
