/* compile with:
What is a makefile :p
/usr/local/cuda/bin/nvcc -std=c++11 glad.c main.cpp kernel.cu sim.cu interop.cu fs_quad.cpp gl_helpers.cpp circle.cpp polygon.cpp cube.cpp -o demo.out -Iinclude -I/usr/local/include -L/usr/local/lib -lglfw3 -lGL -lX11 -lpthread -ldl
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
#include "circle.h"
#include "polygon.h"
#include "cube.h"

GLFWwindow	* p_win;
GLFWwindow 	* p_win2;

GLuint		m_tex;
GLuint		m_prg;

Simulation	* p_sim;
Demo		* p_demo;
CircleDemo	* p_cir;
PolygonDemo * p_poly;
CubeDemo	* p_cube;


int initialize();

void update(double);

void draw(double);

int shutdown();

void initializeDemos();

void destroyDemos();

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
		
		glfwMakeContextCurrent(p_win);
		draw(GLFW_TIME);
        glfwSwapBuffers(p_win);

		if(p_win2){
			glfwMakeContextCurrent(p_win2);
			draw(GLFW_TIME);
			glfwSwapBuffers(p_win2);
		}
    }

    exit(shutdown());
}

float 	m_amp = 1.0f;
int		m_point_count = 3;

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
	else if(key == GLFW_KEY_EQUAL && action == GLFW_PRESS)
		m_amp += 0.1f;
	else if(key == GLFW_KEY_MINUS && action == GLFW_PRESS)
		m_amp -= 0.1f;
	else if(key == GLFW_KEY_1 && action == GLFW_PRESS)
		p_demo = p_cir;
	else if(key == GLFW_KEY_2 && action == GLFW_PRESS)
		p_demo = p_poly;
	else if(key == GLFW_KEY_3 && action == GLFW_PRESS)
		p_demo = p_cube;

	p_demo->KeyPress(key, scancode, action, mods);
}

void initializeDemos()
{
	p_cir = new CircleDemo(PATTERN_HEIGHT * 0.4f, 3);
	p_poly = new PolygonDemo(PATTERN_HEIGHT * 0.4f, 3, 3);
	p_cube = new CubeDemo(PATTERN_HEIGHT * 0.4f, 4);

	p_demo = p_cir;
}

void destroyDemos()
{
	delete(p_cir);
	delete(p_poly);
	delete(p_cube);
}

int initialize()
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {
        return EXIT_FAILURE;
	}

	int monitor_count;
	GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);

	p_win = glfwCreateWindow(ARRAY_HEIGHT * (PATTERN_WIDTH / PATTERN_HEIGHT), ARRAY_HEIGHT, "GPU Holography", NULL, NULL);

	if(monitor_count > 1)
	{
		const GLFWvidmode* mode = glfwGetVideoMode(monitors[1]);
		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

		p_win2 = glfwCreateWindow(mode->width, mode->height, "DMD View", monitors[1], p_win);
	} else {
		p_win2 = nullptr;
	}

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

	initializeDemos();

	return 0;
}

int shutdown()
{
	fsQuadDestroy();

    glfwDestroyWindow(p_win);

	if(p_win2) {
		glfwDestroyWindow(p_win2);
	}

	glDeleteProgram(m_prg);
    glfwTerminate();

	delete(p_sim);
	destroyDemos();

	return EXIT_SUCCESS;
}

void update(double time)
{
	p_sim->SetAmplification(m_amp);
	p_demo->Update(p_sim, time);
}

void draw(double)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, p_sim->GetGLTex());
	glUseProgram(m_prg);

	fsQuadDraw();
}


