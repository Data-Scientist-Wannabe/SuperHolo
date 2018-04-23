#pragma once

#include <math.h>
#include <stdint.h>
#include "SIMD.h"
#include "vector_f.h"

namespace Ruined
{
	namespace Math
	{
		struct ALIGN(16) Matrix
		{
			union
			{
				float	m[4][4];	// [Row][Column]
				float4	r[4];		// Rows

				struct
				{
					float	m00,	m01,	m02,	m03,	// Right
							m10,	m11,	m12,	m13,	// Up
							m20,	m21,	m22,	m23,	// Backward
							m30,	m31,	m32,	m33;	// Translation
				};

			};
			
			Matrix (void);

			Matrix (float const m00, float const m01, float const m02, float const m03,
					float const m10, float const m11, float const m12, float const m13,
					float const m20, float const m21, float const m22, float const m23,
					float const m30, float const m31, float const m32, float const m33);

			Matrix (const float * const pArray);
			
			bool		operator== (const Matrix & other) const;
			bool		operator!= (const Matrix & other) const;

			Matrix &	operator= (const Matrix & other);

			Matrix		operator+ () const;
			Matrix		operator- () const;

			Matrix &	operator+= (const Matrix & other);
			Matrix &	operator-= (const Matrix & other);
			Matrix &	operator*= (const Matrix & other);

			Matrix &	operator*= (const float value);
			Matrix &	operator/= (const float value);

			static Matrix	CreateLookAt(Vector3f const cameraPosition, Vector3f const cameraTarget, Vector3f const cameraUp);
			static void		CreateLookAt(Vector3f const cameraPosition, Vector3f const cameraTarget, Vector3f const cameraUp, Matrix & result);

			static Matrix	CreateOrthographic(float const width, float const height, float const nearZ, float const farZ);
			static void		CreateOrthographic(float const width, float const height, float const nearZ, float const farZ, Matrix & result);

			static Matrix	CreatePerspective(float const width, const float height, float const nearZ, float const farZ);
			static void		CreatePerspective(float const width, const float height, float const nearZ, float const farZ, Matrix & result);

			static Matrix	CreatePerspectiveFov(float const fovAngle, float const aspectRatio, float const nearZ, float const farZ);
			static void		CreatePerspectiveFov(float const fovAngle, float const aspectRatio, float const nearZ, float const farZ, Matrix & result);

			static Matrix	CreateRotationX(float const radians);
			static Matrix	CreateRotationY(float const radians);
			static Matrix	CreateRotationZ(float const radians);
			static void		CreateRotationX(float const radians, Matrix & result);
			static void		CreateRotationY(float const radians, Matrix & result);
			static void		CreateRotationZ(float const radians, Matrix & result);

			static Matrix	CreateScale(float const scale);
			static Matrix	CreateScale(float3 const scales);
			static Matrix	CreateScale(float const xScale, float const yScale, float const zScale);
			static void		CreateScale(float const scale, Matrix & result);
			static void		CreateScale(float3 const & scales, Matrix & result);
			static void		CreateScale(float const xScale, float const yScale, float const zScale, Matrix & result);

			static Matrix	CreateTranslation(float3 const position);
			static Matrix	CreateTranslation(float const xPosition, float const yPosition, float const zPosition);
			static void		CreateTranslation(float3 const & position, Matrix & result);
			static void		CreateTranslation(float const xPosition, float const yPosition, float const zPosition, Matrix & result);
			

			// Returns the identity Matrix
			static Matrix	Identity(void);

			float3 &		Right(void)			{ return reinterpret_cast<float3&>(m00); }
			float3 &		Up(void)			{ return reinterpret_cast<float3&>(m10); }
			float3 &		Backward(void)		{ return reinterpret_cast<float3&>(m20); }
			float3 &		Translation(void)	{ return reinterpret_cast<float3&>(m30); }

			static Matrix	Transpose(Matrix & matrix);
			static void		Transpose(Matrix const & matrix, Matrix & result);	
		};

		inline Matrix operator+ (const Matrix & a, const Matrix & b);
		inline Matrix operator- (const Matrix & a, const Matrix & b);
		inline Matrix operator* (const Matrix & a, const Matrix & b);
		
		inline Matrix operator* (const Matrix & m, const float value);
		inline Matrix operator* (const float value, const Matrix & m);
		inline Matrix operator/ (const Matrix & m, const float value);
	}
}

#include "matrix.hpp"