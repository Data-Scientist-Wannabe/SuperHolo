#include "cube.h"
#include <cmath>
#include <ruined_math/matrix.h>
#include <GLFW/glfw3.h>

#include "dmd.h"

#define CUBE_DEPTH    0.25f
#define CUBE_X        (PATTERN_WIDTH  * 0.5f)
#define CUBE_Y        (PATTERN_HEIGHT * 0.5f)

#define TWO_PI         6.28318530718f

#define CUBE_EDGES     12

void fillPoints(Ruined::Math::Vector4f * points,
                int point_count,
                Ruined::Math::Vector4f start_pt,
                Ruined::Math::Vector4f end_pt)
{
    using namespace Ruined::Math;

    float step = 1.0f / point_count;

    for(int i = 0; i < point_count; i++)
    {
        points[i] = Vector4f::Lerp(start_pt, end_pt, i * step);
    }
}

void CubeDemo::generatePoints()
{
    using namespace Ruined::Math;

    free(this->points);

    this->points = (Vector4 *) malloc(sizeof(Vector4) * CUBE_EDGES * point_count);

    float power = 1.0f / (point_count * CUBE_EDGES);

    // Bottom
    fillPoints(points + 0  * point_count, point_count, Vector4f( 1.0f,-1.0f, 1.0f, power), Vector4f( 1.0f,-1.0f,-1.0f, power));
    fillPoints(points + 1  * point_count, point_count, Vector4f( 1.0f,-1.0f,-1.0f, power), Vector4f(-1.0f,-1.0f,-1.0f, power));
    fillPoints(points + 2  * point_count, point_count, Vector4f(-1.0f,-1.0f,-1.0f, power), Vector4f(-1.0f,-1.0f, 1.0f, power));
    fillPoints(points + 3  * point_count, point_count, Vector4f(-1.0f,-1.0f, 1.0f, power), Vector4f( 1.0f,-1.0f, 1.0f, power));

    // Top
    fillPoints(points + 4  * point_count, point_count, Vector4f( 1.0f, 1.0f, 1.0f, power), Vector4f( 1.0f, 1.0f,-1.0f, power));
    fillPoints(points + 5  * point_count, point_count, Vector4f( 1.0f, 1.0f,-1.0f, power), Vector4f(-1.0f, 1.0f,-1.0f, power));
    fillPoints(points + 6  * point_count, point_count, Vector4f(-1.0f, 1.0f,-1.0f, power), Vector4f(-1.0f, 1.0f, 1.0f, power));
    fillPoints(points + 7  * point_count, point_count, Vector4f(-1.0f, 1.0f, 1.0f, power), Vector4f( 1.0f, 1.0f, 1.0f, power));

    // Sides
    fillPoints(points + 9  * point_count, point_count, Vector4f( 1.0f,-1.0f, 1.0f, power), Vector4f( 1.0f, 1.0f, 1.0f, power));
    fillPoints(points + 9  * point_count, point_count, Vector4f( 1.0f, 1.0f,-1.0f, power), Vector4f( 1.0f,-1.0f,-1.0f, power));
    fillPoints(points + 10 * point_count, point_count, Vector4f(-1.0f,-1.0f,-1.0f, power), Vector4f(-1.0f, 1.0f,-1.0f, power));
    fillPoints(points + 11 * point_count, point_count, Vector4f(-1.0f, 1.0f, 1.0f, power), Vector4f(-1.0f,-1.0f, 1.0f, power));
}

CubeDemo::CubeDemo(float radius, int point_count)
: point_count(point_count), radius(radius), points(nullptr), playing(true), last_time(0.0)
{
    velocity = Ruined::Math::Vector3f::Zero();
    times = Ruined::Math::Vector3f::Zero();
    generatePoints();
}

CubeDemo::~CubeDemo()
{
    free(this->points);
    this->points = nullptr;
}

void CubeDemo::Update(Simulation * sim, double time)
{
    using namespace Ruined::Math;

    float delta = (playing ? time - last_time : 0.0f);

    times += velocity * delta;

    Matrix mat =    Matrix::CreateTranslation(CUBE_X, CUBE_Y, CUBE_DEPTH) *
                    Matrix::CreateRotationZ(times.y ) *
                    Matrix::CreateRotationX(times.x) *
                    Matrix::CreateScale(radius * cosf(times.z) * 3.5f);
    sim->SetTransformation(mat);

    sim->setPoints(this->points, this->point_count * CUBE_EDGES);

    sim->generateImage();

    last_time = time;
}

void CubeDemo::KeyPress(int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_UP && action == GLFW_PRESS){
        point_count++;
		generatePoints();
	} else if(key == GLFW_KEY_DOWN && action == GLFW_PRESS){  
        point_count = (point_count <= 1 ? 1 : point_count - 1);
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