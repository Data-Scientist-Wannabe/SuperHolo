#pragma once

#include "sim.h"

class Demo
{
public:
    virtual void KeyPress(int key, int scancode, int action, int mods) {}
    virtual void Update(Simulation * sim, double time) {}
};