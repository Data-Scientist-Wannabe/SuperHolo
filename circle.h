#include <ruined_math/vector_f.h>
#include "sim.h"

class CircleDemo
{
    Ruined::Math::Vector4 * points;
    int point_count;
    float radius;

public:
    CircleDemo(float radius, int points);
    ~CircleDemo();

    void UpdateSim(Simulation * sim, double time);
    int PointCount(void) {return point_count;}
};