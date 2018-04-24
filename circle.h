#include <ruined_math/vector_f.h>
#include "../demo.h"

class CircleDemo : public Demo
{
    Ruined::Math::Vector4 * points;
    int point_count;
    float radius;
    bool playing;
    double timer;

protected:
    void generatePoints();

public:
    CircleDemo(float radius, int points);
    ~CircleDemo();

    void Update(Simulation * sim, double time);
    void KeyPress(int key, int scancode, int action, int mods);
};