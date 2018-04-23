#include "circle.h"
#include <cmath>
#include <ruined_math/matrix.h>

#include "dmd.h"

#define CIRCLE_DEPTH    0.33f
#define CIRCLE_X        (PATTERN_WIDTH  * 0.5f)
#define CIRCLE_Y        (PATTERN_HEIGHT * 0.5f)

#define TWO_PI          6.28318530718f

CircleDemo::CircleDemo(float radius, int point_count)
: point_count(point_count), radius(radius)
{
    using namespace Ruined::Math;

    this->points = (Vector4 *) malloc(sizeof(Vector4) * point_count);

    float delta = TWO_PI / point_count;
    float power = 1.0f / point_count;

    for(int i = 0; i < point_count; i++)
    {
        this->points[i] = Vector4(sinf(delta * i), cosf(delta * i), 0.0f, power);
    }
}

CircleDemo::~CircleDemo()
{
    free(this->points);
    this->points = nullptr;
}

void CircleDemo::UpdateSim(Simulation * sim, double time)
{
    using namespace Ruined::Math;
    Matrix mat =    Matrix::CreateTranslation(CIRCLE_X, CIRCLE_Y, CIRCLE_DEPTH) *
                    Matrix::CreateRotationZ(time) *
                    //Matrix::CreateRotationX(time * 0.9f) *
                    Matrix::CreateScale(radius);
    sim->SetTransformation(mat);
    sim->setPoints(this->points, this->point_count);
}