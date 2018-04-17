// compile with g++ -std=c++11 glad.c glfw.cpp -o glfw.out -Iinclude -I/usr/local/include -L/usr/local/lib -lglfw3 -lGL -lX11 -lpthread -ldl
#include <glad/glad.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>

#define TEXTURE_SIZE	128

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cout << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

void * cuda_dev_render_buffer;
struct cudaGraphicsResource * 		cuda_tex_resource;

static const GLfloat g_vertex_buffer_data[] = {
/*	position x, y	texcoord u,v */
   -1.0f, -1.0f,	0.0f, 1.0f, 
   -1.0f,  1.0f,	0.0f, 0.0f,
    1.0f, -1.0f,	1.0f, 1.0f,
	1.0f,  1.0f,	1.0f, 0.0f,
};

__global__ void drawCuda(float * values)
{
    for(int x = threadIdx.x; x < TEXTURE_SIZE; x += blockDim.x)
    {
        for(int y = blockIdx.x; y < TEXTURE_SIZE; y+= gridDim.x)
        {
			uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
    		values[x + y * TEXTURE_SIZE] = ((x & 0x20) ? 1.0f : 0.5f) * ((y & 0x20) ? 1.0f : 0.5f);
		}
	}
}

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

void generateCUDAImage()
{
	dim3 dimGrid(TEXTURE_SIZE); 	// N_BLOCKS
	dim3 dimBlock(TEXTURE_SIZE); 	// N_THREADS
	drawCuda<<<dimGrid, dimBlock>>>((float*)cuda_dev_render_buffer);
	
	cudaArray * texture_ptr;
	CUDA_CALL(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
	CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

	int size_tex_data = TEXTURE_SIZE * TEXTURE_SIZE * sizeof(float);

	CUDA_CALL(cudaMemcpyToArray(
		texture_ptr, 0, 0, 
		cuda_dev_render_buffer, 
		size_tex_data, 
		cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
}

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path){

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if(VertexShaderStream.is_open()){
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	}else{
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if(FragmentShaderStream.is_open()){
		std::stringstream sstr;
		sstr << FragmentShaderStream.rdbuf();
		FragmentShaderCode = sstr.str();
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;


	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}



	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}



	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> ProgramErrorMessage(InfoLogLength+1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}

	
	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);
	
	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

int main(void)
{
    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);
    window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
    glfwSetKeyCallback(window, key_callback);
    printf("%s\n", glGetString(GL_VERSION));

    // This will identify our vertex buffer
    GLuint vertexbuffer;
    // Generate 1 buffer, put the resulting identifier in vertexbuffer
    glGenBuffers(1, &vertexbuffer);
    // The following commands will talk about our 'vertexbuffer' buffer
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    // Give our vertices to OpenGL.
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    GLuint programID = LoadShaders( "shaders/test_vs.glsl", "shaders/test_ps.glsl" );

	GLuint texture;
	createGLTextureForCUDA(&texture, &cuda_tex_resource, TEXTURE_SIZE, TEXTURE_SIZE);
	cudaMalloc(&cuda_dev_render_buffer, TEXTURE_SIZE * TEXTURE_SIZE * sizeof(float));

	
	
	glUniform1i(glGetUniformLocation(programID, "tex"), texture);

    while (!glfwWindowShouldClose(window))
    {
		generateCUDAImage();
		glfwPollEvents();

        glClear(GL_COLOR_BUFFER_BIT |  GL_DEPTH_BUFFER_BIT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);

		glUseProgram(programID);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0 );
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        
		// Draw the triangle !
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray(0);

        glfwSwapBuffers(window);      
    }

	cudaGraphicsUnregisterResource(cuda_tex_resource);

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
