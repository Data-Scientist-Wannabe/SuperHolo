#include "interop.h"

// Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
void createGLTextureForCUDA(
	GLuint * gl_tex,
	cudaGraphicsResource** cuda_tex, 
	unsigned int size_x,
	unsigned int size_y)
{
	glGenTextures(1, gl_tex);
	glBindTexture(GL_TEXTURE_2D, *gl_tex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, size_x, size_y, 0, GL_RED, GL_FLOAT, NULL);

	CUDA_CALL(cudaGraphicsGLRegisterImage(
		cuda_tex,
		*gl_tex,
		GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsWriteDiscard));
}