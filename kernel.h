#include <cuda.h>
#include "dmd.h"

#define CUDART_2PI      6.2831853071795865f
#define CUDART_PI       3.1415926535897931f
#define CUDART_PIO6     0.52359877559829887f

#define LAMBDA          632.8e-9f
#define LAMBDA_INV      1.580278128950695e6f
#define TWO_LAMBDA_INV  3.160556257901391e6f
#define REF_BEAM_ANGLE  (0.00565003f)    // ~0.33 degrees

#define PLANE_CONST 57187.65f
#define VAL_CONST   9929180.296f

extern "C" void launch_kernel( dim3 grid, dim3 block, float * pattern, float4 * points, int count);

extern "C" void launch_transform( dim3 grid, dim3 block, float4 * points_in, float4 * points_out,
                    float4 * matrix, float amp, int count);
