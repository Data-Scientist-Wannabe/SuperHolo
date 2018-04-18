/* compile with:
/usr/local/cuda/bin/nvcc -std=c++11 glad.c glfw.cpp kernel.cu kinect.cpp sim.cu interop.cu -o realtime.out -Iinclude -I/usr/local/include -L/usr/local/lib -lglfw3 -lGL -lX11 -lpthread -ldl -lfreenect
*/
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

#include "kinect.h"
#include "sim.h"
#include "dmd.h"

static const GLfloat g_vertex_buffer_data[] = {
/*	position x, y	texcoord u,v */
   -1.0f, -1.0f,	0.0f, 1.0f, 
   -1.0f,  1.0f,	0.0f, 0.0f,
    1.0f, -1.0f,	1.0f, 1.0f,
	1.0f,  1.0f,	1.0f, 0.0f,
};

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

int initializeFreenect(struct Kinect * kinect, int index)
{
	freenect_context * fn_ctx;
	int ret = freenect_init(&fn_ctx, NULL);
	if (ret < 0)
		return ret;

	// Show debug messages and use camera only.
	freenect_set_log_level(fn_ctx, FREENECT_LOG_DEBUG);
	freenect_select_subdevices(fn_ctx, FREENECT_DEVICE_CAMERA);

	// Show debug messages and use camera only.
	freenect_set_log_level(fn_ctx, FREENECT_LOG_DEBUG);
	freenect_select_subdevices(fn_ctx, FREENECT_DEVICE_CAMERA);
	// Find out how many devices are connected.
	int num_devices = ret = freenect_num_devices(fn_ctx);
	if (ret < 0)
		return ret;

	if (num_devices == 0)
	{
		printf("No device found!\n");
		freenect_shutdown(fn_ctx);
		return 1;
	}

	ret = kinect_init(kinect, fn_ctx, index);
	if (ret < 0)
	{
		freenect_shutdown(fn_ctx);
		return ret;
	}

	printf("freenect initialized\n");
	return 0;
}

int startKinect(struct Kinect * kinect, pthread_t * thread)
{
	if (int ret = pthread_create(thread, NULL, kinect_threadfunc, (void*)kinect)) {
		printf("pthread_create failed\n");
		kinect_destroy(kinect);
		freenect_shutdown(kinect->fn_ctx);
		return ret;
	}

	printf("kinect started\n");
	return 0;
}

int main(void)
{
	int ret;

    GLFWwindow* window;

	pthread_t kinect_thread;

	struct Kinect kinect;

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

	Simulation sim(ARRAY_WIDTH, ARRAY_HEIGHT, KINECT_WIDTH * KINECT_HEIGHT);
	
	if( ret = initializeFreenect(&kinect, 0)){
		return ret; }

	if( ret = startKinect(&kinect, &kinect_thread)) {
		return ret; }

    // This will identify our vertex buffer
    GLuint vertexbuffer;
    // Generate 1 buffer, put the resulting identifier in vertexbuffer
    glGenBuffers(1, &vertexbuffer);
    // The following commands will talk about our 'vertexbuffer' buffer
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    // Give our vertices to OpenGL.
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    GLuint programID = LoadShaders( "shaders/test_vs.glsl", "shaders/test_ps.glsl" );
	glUniform1i(glGetUniformLocation(programID, "tex"), sim.GetGLTex());

	uint32_t old_time = 0;

    while (!glfwWindowShouldClose(window))
    {
		if(kinect.depth_timestamp != old_time)
		{
			old_time = kinect.depth_timestamp;
			sim.setPoints(&kinect, 8000);
			sim.generateImage(4);
		}
		glfwPollEvents();

        glClear(GL_COLOR_BUFFER_BIT |  GL_DEPTH_BUFFER_BIT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, sim.GetGLTex());

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

	kinect.running = false;
	pthread_join(kinect_thread, NULL);
	kinect_destroy(&kinect);
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
