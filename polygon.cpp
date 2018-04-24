#include "polygon.h"
#include <cmath>
#include <ruined_math/matrix.h>
#include <GLFW/glfw3.h>

#include "dmd.h"

#define POLYGON_DEPTH    0.25f
#define POLYGON_X        (PATTERN_WIDTH  * 0.5f)
#define POLYGON_Y        (PATTERN_HEIGHT * 0.5f)

#define TWO_PI          6.28318530718f

void PolygonDemo::generatePoints()
{
    using namespace Ruined::Math;

    free(this->points);

    this->points = (Vector4 *) malloc(sizeof(Vector4) * side_count * point_count);

    float delta = TWO_PI / side_count;
    float step = 1.0f / point_count;
    float power = 1.0f / (point_count * side_count);

    for(int s = 0; s < side_count; s++)
    {
        Vector4 start_pt   = Vector4(cosf(delta * s), sinf(delta * s), 0.0f, power);
        Vector4 end_pt     = Vector4(cosf(delta * (s + 1)), sinf(delta * (s + 1)), 0.0f, power);

        for(int p = 0; p < point_count; p++)
        {
            this->points[s * point_count + p] = Vector4f::Lerp(start_pt, end_pt, p * step);
        }
    }
}

PolygonDemo::PolygonDemo(float radius, int side_count, int point_count)
: point_count(point_count), side_count(side_count), radius(radius), points(nullptr), playing(true), last_time(0.0)
{
    velocity = Ruined::Math::Vector3f::Zero();
    times = Ruined::Math::Vector3f::Zero();
    generatePoints();
}

PolygonDemo::~PolygonDemo()
{
    free(this->points);
    this->points = nullptr;
}

void PolygonDemo::Update(Simulation * sim, double time)
{
    using namespace Ruined::Math;

    float delta = (playing ? time - last_time : 0.0f);

    times += velocity * delta;

    Matrix mat =    Matrix::CreateTranslation(POLYGON_X, POLYGON_Y, POLYGON_DEPTH) *
                    Matrix::CreateRotationZ(times.y ) *
                    Matrix::CreateRotationX(times.x) *
                    Matrix::CreateScale(radius * cosf(times.z) * 3.5f);
    sim->SetTransformation(mat);

    sim->setPoints(this->points, this->point_count * this->side_count);

    sim->generateImage();

    last_time = time;
}

void PolygonDemo::KeyPress(int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_UP && action == GLFW_PRESS){
        point_count++;
		generatePoints();
	} else if(key == GLFW_KEY_DOWN && action == GLFW_PRESS){  
        point_count = (point_count <= 0 ? 0 : point_count - 1);
		generatePoints();
	} else if(key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
        side_count++;
		generatePoints();
	} else if(key == GLFW_KEY_LEFT && action == GLFW_PRESS){  
        side_count = (side_count <= 3 ? 3 : side_count - 1);
		generatePoints();
	}else if(key == GLFW_KEY_W && action == GLFW_PRESS) {
        velocity.x += 0.2f;
    }else if(key == GLFW_KEY_S && action == GLFW_PRESS) {
        velocity.x -= 0.2f;
    }else if(key == GLFW_KEY_D && action == GLFW_PRESS) {
        velocity.y += 0.2f;
    }else if(key == GLFW_KEY_A && action == GLFW_PRESS) {
        velocity.y -= 0.2f;
    }else if(key == GLFW_KEY_E && action == GLFW_PRESS) {
        velocity.z += 0.2f;
    }else if(key == GLFW_KEY_Q && action == GLFW_PRESS) {
        velocity.z -= 0.2f;
    }else if(key == GLFW_KEY_R && action == GLFW_PRESS) {
        if(velocity.LengthSquared() > 0.01f)
            velocity = Ruined::Math::Vector3::Zero();
        else
            times = Ruined::Math::Vector3f::Zero();
    }

}