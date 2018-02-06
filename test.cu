#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define ARRAY_SIZE 16

#define N_BLOCKS 1
#define N_THREADS 128

__global__ void array_doubler(float * array, int size)
{
    for(int i = threadId.x; i < size; i += blockDim.x)
    {
        array[i] = array[i] * 2;
    }

    return;
}

int main(void)
{
    float * h_array;
    float * d_array;

    dim3 dimGrid(N_BLOCKS);
    dim3 dimBlock(N_THREADS);

    h_array = (float*)malloc(ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&d_array, ARRAY_SIZE * sizeof(float));


    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        h_array[i] = (float)i;
    }

    cudaMemcpy(d_array, h_array, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // call method
    array_doubler<<<dimGrid, dimBlock>>>(d_array, ARRAY_SIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(h_array, d_array, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        printf("%f\n", h_array[i]);
    }

    free(h_array);
    cudaFree(d_array);

    return 0;
}