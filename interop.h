#include <glad/glad.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cout << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

extern void createGLTextureForCUDA(
	GLuint * gl_tex,
	cudaGraphicsResource** cuda_tex, 
	unsigned int size_x,
	unsigned int size_y);