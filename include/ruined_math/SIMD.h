#pragma once

#include <x86intrin.h>

#define RUINED_SSE		10
#define RUINED_SSE2		20
#define RUINED_SSE3		30
#define __RUINED_SSE__ 	RUINED_SSE3

//#define __NEON__

// #define __SIMD__ __SSE__

#ifdef _MSC_VER
#define ALIGN( n )	__declspec( align( n ) )
#else
#define ALIGN( n ) alignas( n )
#endif

namespace Ruined
{
	namespace Math
	{
		struct SIMD
		{
			typedef __m128		vFloat128;
			typedef __m128d		vDouble128;
			typedef __m128i		vInt128;

			//typedef __m256		vFloat256;
			//typedef __m256d		vDouble256;
			//typedef __m256i		vInt256;

			typedef vFloat128		vFloat;
			typedef vDouble128		vDouble;
			typedef vInt128			vInt;

		#pragma region VFLOAT

		#pragma region ARITHMATIC
				// Adds the four floating-point values of A and B :
				//	returns (A.x + B.x, A.y + B.y, A.z + B.z, A.w + B.w)
			static	vFloat	Add			(vFloat A, vFloat B) { return _mm_add_ps(A, B); }
				// Adds adjacent elements of A and B :
				//	returns (A.x + A.y, A.z + A.w, B.x + B.y, B.z + B.w)
			static	vFloat	AddH		(vFloat A, vFloat B) { return _mm_hadd_ps(A, B); }
				// Adds the lower floating-point value of A and B and copies the upper three from A :
				//	returns (A.x + B.x, A.y, A.z, A.w)
			static	vFloat	AddS		(vFloat A, vFloat B) { return _mm_add_ss(A, B); }
				
				// Subtracts the four floating-point values of A by B :
				//	returns (A.x - B.x, A.y - B.y, A.z - B.z, A.w - B.w)
			static	vFloat	Subtract	(vFloat A, vFloat B) { return _mm_sub_ps(A, B); }
				// Subtracts adjacent elements of A and B :
				//	returns (A.x - A.y, A.z - A.w, B.x - B.y, B.z - B.w)
			static	vFloat	SubtractH	(vFloat A, vFloat B) { return _mm_hsub_ps(A, B); }
				// Subtracts the lower floating-point value of A by B and copies the upper three from A :
				//	returns (A.x - B.x, A.y, A.z, A.w)
			static	vFloat	SubtractS	(vFloat A, vFloat B) { return _mm_sub_ss(A, B); }
				
				// Multiplies the four floating-point values of A and B :
				//	returns (A.x * B.x, A.y * B.y, A.z * B.z, A.w * B.w)
			static	vFloat	Multiply	(vFloat A, vFloat B) { return _mm_mul_ps(A, B); }
				// Multiplies the lower floating-point value of A and B and copies the upper three from A :
				//	returns (A.x * B.x, A.y, A.z, A.w)
			static	vFloat	MultiplyS	(vFloat A, vFloat B) { return _mm_mul_ss(A, B); }

				// Divides the four floating-point values of A by B :
				//	returns (A.x / B.x, A.y / B.y, A.z / B.z, A.w / B.w)
			static	vFloat	Divide		(vFloat A, vFloat B) { return _mm_div_ps(A, B); }
				// Divides the lower floating-point value of A by B and copies the upper three from A :
				//	returns (A.x / B.x, A.y, A.z, A.w)
			static	vFloat	DivideS		(vFloat A, vFloat B) { return _mm_div_ss(A, B); }
				
				// Determines the maximum four floating-point values of A and B :
				//	returns (max(A.x, B.x), max(A.y, B.y), max(A.z, B.z), max(A.w, B.w)
			static	vFloat	Max			(vFloat A, vFloat B) { return _mm_max_ps(A, B); }
				// Determines the maximum lower floating-point value of A and B and copies the upper three from A :
				//	returns (max(A.x, B.x), A.y, A.z, A.w)
			static	vFloat	MaxS		(vFloat A, vFloat B) { return _mm_max_ss(A, B); }

				// Determines the minimum four floating-point values of A and B :
				//	returns (min(A.x, B.x), min(A.y, B.y), min(A.z, B.z), min(A.w, B.w)
			static	vFloat	Min			(vFloat A, vFloat B) { return _mm_min_ps(A, B); }
				// Determines the minimum lower floating-point value of A and B and copies the upper three from A :
				//	returns (min(A.x, B.x), A.y, A.z, A.w)
			static	vFloat	MinS		(vFloat A, vFloat B) { return _mm_min_ss(A, B); }

				// Calculates the square roots of the four floating-point values of A :
				//	returns (sqrt(A.x), sqrt(A.y), sqrt(A.z), sqrt(A.w))
			static	vFloat	Sqrt		(vFloat A) { return _mm_sqrt_ps(A); }
				// Calculates the square root of the lower floating-point value of A and copies the upper three :
				//	returns (sqrt(A.x), A.y, A.z, A.w)
			static	vFloat	SqrtS		(vFloat A) { return _mm_sqrt_ss(A); }
				
				// Calculates the reciprocals of the four floating-point values of A :
				//	returns (1/A.x, 1/A.y, 1/A.z, 1/A.w)
			static	vFloat	Reciprocate	(vFloat A) { return _mm_rcp_ps(A); }
				// Calculates the reciprocal of the lower floating-point value of A and copies the upper three :
				//	returns (1/A.x, A.y, A.z, A.w)
			static	vFloat	ReciprocateS(vFloat A) { return _mm_rcp_ss(A); }

				// Approximates the reciprocals of the square roots of the four floating-point values of A :
				//	returns (1/sqrt(A.x), 1/sqrt(A.y), 1/sqrt(A.z), 1/sqrt(A.w))
			static	vFloat	ReciprocalSqrt	(vFloat A) { return _mm_rsqrt_ps(A); }
				// Approximates the reciprocal of the square toot of the lower floating-point value of A and copies the upper three :
				//	returns (1/sqrt(A.x), A.y, A.z, A.w)
			static	vFloat	ReciprocalSqrtS(vFloat A) { return _mm_rsqrt_ss(A); }
		#pragma endregion
		#pragma region LOAD
				// Loads four floating-point values. Note: these values must be aligned on a 16byte address :
				//	returns (p[0], p[1], p[2], p[3])
			static	vFloat	Load		(const float * p) { return _mm_load_ps(p); }
				// Loads four floating-point values :
				//	returns (p[0], p[1], p[2], p[3])
			static	vFloat	LoadU		(const float * p) { return _mm_loadu_ps(p); }
				// Loads a single floating-point value into the low word and clears the upper three words :
				//	returns (p[0], 0, 0, 0)
			static	vFloat	LoadS		(const float * p) { return _mm_load_ss(p); }
				// Loads a single floating-point value, copying it into all four words :
				//	returns (p[0], p[0], p[0], p[0])
			static	vFloat	Load1		(const float * p) { return _mm_load_ps1(p); }
		#pragma endregion
		#pragma region SET
				// Sets the four floating-point values to the four inputs :
				//	returns (a, b, c, d)
			static	vFloat	Set			(const float a, const float b, const float c, const float d) { return _mm_set_ps(a, b, c, d); }
				// Sets the low word of a vFloat to w and clears the upper three words :
				//	returns (w, 0, 0, 0)
			static	vFloat	SetS		(const float w) { return _mm_set_ss(w); }
				// Sets the four floating-point values to w :
				//	returns (w, w, w, w)
			static	vFloat	Set1		(const float w) { return _mm_set1_ps(w); }
				// Clears all four values
				//	returns (0, 0, 0, 0)
			static	vFloat	SetZero		(void) { return _mm_setzero_ps(); }
		#pragma endregion		
		#pragma region STORE
				// Stores four floating-point values.
			static	void	Store		(float * p, vFloat v) { _mm_store_ps(p, v); }
				// Stores the lower floating-point value.
			static	void	StoreS		(float * p, vFloat v) { _mm_store_ss(p, v); }
		#pragma endregion
		#pragma region MISC
				// Returns the x value of a vFloat.
			static	float	GetX		(vFloat v) 
			{ 
				return _mm_cvtss_f32(v);
			}
				// Returns the y value of a vFloat.
			static	float	GetY		(vFloat v) 
			{ 
				vFloat t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
				return _mm_cvtss_f32(t);
			}
				// Returns the z value of a vFloat.
			static	float	GetZ		(vFloat v) 
			{ 
				vFloat t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
				return _mm_cvtss_f32(t); 
			}
				// Returns the w value of a vFloat.
			static	float	GetW		(vFloat v) 
			{ 
				vFloat t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
				return _mm_cvtss_f32(t); 
			}

				// Unpacks the upper half of A and B
				//	returns (A.z, B.z, A.w, B.w)
			static	vFloat	UnpackHigh	(vFloat A, vFloat B) { return _mm_unpackhi_ps(A, B); }
				// Unpacks the lowwer half of A and B
				//	returns (A.x, B.x, A.y, B.y)
			static	vFloat	UnpackLow	(vFloat A, vFloat B) { return _mm_unpacklo_ps(A, B); }
				// Selects four floating-point values from A and B, based on the mask m.
			#define		SIMD_SHUFFLE_PS( A , B , m ) _mm_shuffle_ps( A , B , m )

			#define		SIMD_REORDER_PS( A , m ) _mm_shuffle_ps( A , A , m )
			//static	inline vFloat	Shuffle		(vFloat A, vFloat B, unsigned int const m) { return _mm_shuffle_ps(A, B, m); }

		#pragma endregion
		#pragma region MOVE
				// Moves the upper of half of B to the lower half of A.
				//	returns (B.z, B.w, A.z, A.w)
			static	vFloat	MoveHighHalf	(vFloat A, vFloat B) { return _mm_movehl_ps(A, B); }	
				// Moves the lower of half of B to the upper half of A.
				//	returns (A.x, A.y, B.x, B.y)
			static	vFloat	MoveLowHalf	(vFloat A, vFloat B) { return _mm_movelh_ps(A, B); }				
		#pragma endregion

		#pragma endregion

			static	vDouble	Add			(vDouble A, vDouble B) { return _mm_add_pd(A, B); }
			static	vDouble	Subtract	(vDouble A, vDouble B) { return _mm_sub_pd(A, B); }
			static	vDouble	Multiply	(vDouble A, vDouble B) { return _mm_mul_pd(A, B); }
			static	vDouble	Divide		(vDouble A, vDouble B) { return _mm_div_pd(A, B); }

			static	vInt	Add			(vInt A, vInt B) { return _mm_add_epi32(A, B); }
			static	vInt	Subtract	(vInt A, vInt B) { return _mm_sub_epi32(A, B); }
			static	vInt	Multiply	(vInt A, vInt B) { return _mm_mul_epi32(A, B); }
		};
	}
}