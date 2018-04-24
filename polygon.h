#include <ruined_math/vector_f.h>
#include "demo.h"

class PolygonDemo : public Demo
{
    Ruined::Math::Vector4 * points;
    int point_count;
    int side_count;
    float radius;
    bool playing;
    double last_time;
    Ruined::Math::Vector3f velocity;
    Ruined::Math::Vector3f times;

protected:
    void generatePoints();

public:
    PolygonDemo(float radius, int side_count, int points);
    ~PolygonDemo();

    void Update(Simulation * sim, double time);
    void KeyPress(int key, int scancode, int action, int mods);
};