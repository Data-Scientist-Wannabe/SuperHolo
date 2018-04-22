/* compile with:
/usr/local/cuda/bin/nvcc -std=c++11 glad.c demo.cpp kernel.cu sim.cu interop.cu fs_quad.cpp gl_helpers.cpp -o realtime.out -Iinclude -I/usr/local/include -L/usr/local/lib -lglfw3 -lGL -lX11 -lpthread -ldl
*/
#include <glad/glad.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>
#include <cstdio>

#include "gl_helpers.h"
#include "sim.h"
#include "dmd.h"
#include "fs_quad.h"

GLFWwindow * p_win;
Simulation * p_sim;
GLuint		 m_tex;
GLuint		 m_prg;

int initialize();

void update(double);

void draw(double);

int shutdown();

int main(void)
{
	double GLFW_TIME;

	if(int ret = initialize()){
		exit(ret);
	}

    while (!glfwWindowShouldClose(p_win))
    {
		glfwPollEvents();
		GLFW_TIME = glfwGetTime();

		update(GLFW_TIME);
		
		draw(GLFW_TIME);
		
        glfwSwapBuffers(p_win);      
    }

    exit(shutdown());
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

int initialize()
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {
        return EXIT_FAILURE;
	}

    p_win = glfwCreateWindow(ARRAY_WIDTH, ARRAY_HEIGHT, "GPU Holography", NULL, NULL);
    if (!p_win){
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(p_win);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
    glfwSetKeyCallback(p_win, key_callback);
    printf("%s\n", glGetString(GL_VERSION));

	p_sim = new Simulation(ARRAY_WIDTH, ARRAY_HEIGHT, 1024);

    m_prg = LoadShaders( "shaders/quad.vs.glsl", "shaders/quad.ps.glsl" );
	glUniform1i(glGetUniformLocation(m_prg, "tex"), p_sim->GetGLTex());

	fsQuadInitialize();

	return 0;
}

int shutdown()
{
	fsQuadDestroy();
    glfwDestroyWindow(p_win);
	glDeleteProgram(m_prg);
    glfwTerminate();

	return EXIT_SUCCESS;
}

float4 p_points[4];

void update(double)
{
	p_points[0] = { PATTERN_WIDTH * 0.33f, PATTERN_HEIGHT * 0.20f, 0.30f, 0.33f };
    p_points[1] = { PATTERN_WIDTH * 0.50f, PATTERN_HEIGHT * 0.40f, 0.30f, 0.33f };
    p_points[2] = { PATTERN_WIDTH * 0.66f, PATTERN_HEIGHT * 0.60f, 0.30f, 0.33f };
	p_points[3] = { PATTERN_WIDTH * 0.50f, PATTERN_HEIGHT * 0.80f, 0.30f, 0.33f };

	p_sim->setPoints(p_points, 4);
	p_sim->generateImage(4);
}
#include <cuda.h>
void draw(double)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, p_sim->GetGLTex());
	glUseProgram(m_prg);

	fsQuadDraw();
}


