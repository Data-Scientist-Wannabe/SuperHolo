#include <cuda.h>
#include "glad/glad.h"
#include "kinect.h"

#define N_BLOCKS    541      //
#define N_THREADS   512     // Threads per block

class Simulation
{
private:
    int width, height;
    int max_point_count;

    float4 * d_points;

    float * d_pattern;
    
    void * cuda_dev_render_buffer;
    struct cudaGraphicsResource * 		cuda_tex_resource;
    
    GLuint gl_tex;

    float scale;
    float translation;

public:
    Simulation(int width, int height, int max_points);

    GLuint GetGLTex(void) { return gl_tex; }

    void SetScale(float s) { this->scale = scale; }
    void SetTranslation(float t) { this->translation = t; }

    void generateImage(int);
    void setPoints(float4 * points, int count);

};