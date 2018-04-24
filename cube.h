#include <ruined_math/vector_f.h>
#include "demo.h"

class CubeDemo : public Demo
{
    Ruined::Math::Vector4 * points;
    int point_count;
    float radius;
    bool playing;
    double last_time;
    Ruined::Math::Vector3f velocity;
    Ruined::Math::Vector3f times;

protected:
    void generatePoints();

public:
    CubeDemo(float radius, int points);
    ~CubeDemo();

    void Update(Simulation * sim, double time);
    void KeyPress(int key, int scancode, int action, int mods);
};