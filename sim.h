#pragma once

#include <cuda.h>
#include <ruined_math/matrix.h>

#include "glad/glad.h"
#include "kinect.h"

#define N_BLOCKS    541      //
#define N_THREADS   512     // Threads per block

class Simulation
{

private:
    int width, height;
    int max_point_count;
    int point_count;

    float * d_points;
    float * d_pattern;
    float * d_transform;
    float * cuda_dev_render_buffer;

    struct cudaGraphicsResource * 		cuda_tex_resource;
    GLuint gl_tex;

    float     amp;

public:
    Simulation(int width, int height, int max_points);
    ~Simulation();

    GLuint GetGLTex(void) { return gl_tex; }

    void SetAmplification(float a) { this->amp = a; }
    void SetTransformation(Ruined::Math::Matrix transform);

    void generateImage(void);
    void setPoints(void * points, int count);

};