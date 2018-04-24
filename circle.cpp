#include "circle.h"
#include <cmath>
#include <ruined_math/matrix.h>
#include <GLFW/glfw3.h>

#include "dmd.h"

#define CIRCLE_DEPTH    0.33f
#define CIRCLE_X        (PATTERN_WIDTH  * 0.5f)
#define CIRCLE_Y        (PATTERN_HEIGHT * 0.5f)

#define TWO_PI          6.28318530718f

void CircleDemo::generatePoints()
{
    using namespace Ruined::Math;

    free(this->points);

    this->points = (Vector4 *) malloc(sizeof(Vector4) * point_count);

    float delta = TWO_PI / point_count;
    float power = 1.0f / point_count;

    for(int i = 0; i < point_count; i++)
    {
        this->points[i] = Vector4(sinf(delta * i), cosf(delta * i), 0.0f, power);
    }
}

CircleDemo::CircleDemo(float radius, int point_count)
: point_count(point_count), radius(radius), points(nullptr), playing(true), timer(0.0)
{
    generatePoints();
}

CircleDemo::~CircleDemo()
{
    free(this->points);
    this->points = nullptr;
}

void CircleDemo::Update(Simulation * sim, double time)
{
    using namespace Ruined::Math;

    if(playing)
        timer = time;

    Matrix mat =    Matrix::CreateTranslation(CIRCLE_X, CIRCLE_Y, CIRCLE_DEPTH) *
                    Matrix::CreateRotationZ(timer) *
                    Matrix::CreateRotationX(timer * 0.9f) *
                    Matrix::CreateScale(radius * sinf(timer * 0.5f) * 4.0f);
    sim->SetTransformation(mat);
    sim->setPoints(this->points, this->point_count);
    sim->generateImage();
}

void CircleDemo::KeyPress(int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_UP && action == GLFW_PRESS)
	{
        point_count++;
		generatePoints();
	} else if(key == GLFW_KEY_DOWN && action == GLFW_PRESS)
	{  
        point_count = (point_count <= 0 ? 0 : point_count - 1);
		generatePoints();
	} else if(key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        playing = !playing;
    }
}