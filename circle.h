#include <ruined_math/vector_f.h>
#include "demo.h"

class CircleDemo : public Demo
{
    Ruined::Math::Vector4 * points;
    int point_count;
    float radius;

public:
    CircleDemo(float radius, int points);
    ~CircleDemo();

    void Update(Simulation * sim, double time);
};