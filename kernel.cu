#include <cuda.h>
#include "kernel.h"

__device__ __forceinline__ float plane(float x)
{
    return __cosf(PLANE_CONST * x);
}

__device__ __forceinline__ float distance(float2 uv, float3 point)
{
    return norm3df(uv.x - point.x, uv.y - point.y, point.z);
}

__device__ __forceinline__ float val(float2 uv, float3 point)
{
    float d = distance(uv, point);
    return __sinf(remainderf(d, LAMBDA) * VAL_CONST);
}

__device__ __forceinline__ float intensity(float2 uv, float3 point)
{
    float x = uv.x - point.x;
    float y = uv.y - point.y;
    return 1.0f / (x * x + y * y + point.z * point.z);
}

__global__ void simulation(float * pattern, float4 * points, int count)
{
    float sum;

    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < ARRAY_SIZE; index += gridDim.x * blockDim.x)
    {
        int y = index / ARRAY_WIDTH;
        int x = index - (y * ARRAY_WIDTH);
        float2 uv = make_float2(x * ARRAY_WIDTH_INV * PATTERN_WIDTH, y * ARRAY_HEIGHT_INV * PATTERN_HEIGHT);

        sum = plane(uv.x);

        for(int i = 0; i < count; i++)
        {
            float3 point = make_float3(points[i].x, points[i].y, points[i].z);
            sum += points[i].w * intensity(uv, point) * val(uv, point);
        }

        pattern[index] = sum;
    }
}

__global__ void transform(float4 * pin, float4 * pout, float4 * mat, float amp, int count)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
    {
        pout[i].x = mat[0].x * pin[i].x + mat[0].y * pin[i].y + mat[0].z * pin[i].z + mat[0].w;
        pout[i].y = mat[1].x * pin[i].x + mat[1].y * pin[i].y + mat[1].z * pin[i].z + mat[1].w;
        pout[i].z = mat[2].x * pin[i].x + mat[2].y * pin[i].y + mat[2].z * pin[i].z + mat[2].w;
        pout[i].w = pin[i].w * amp;
    }
}

extern "C" void
launch_kernel(  dim3 grid, 
                    dim3 block,
                    float * pattern,
                    float4 * points,
                    int count)
{
    simulation<<<grid, block>>>(pattern, points, count);
}

extern "C" void
launch_transform(   dim3 grid, 
                    dim3 block,
                    float4 * points_in,
                    float4 * points_out,
                    float4 * matrix,
                    float amp,
                    int count)
{
    transform<<<grid, block>>>(points_in, points_out, matrix, amp, count);
}