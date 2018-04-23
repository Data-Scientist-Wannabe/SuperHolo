#pragma once

#include <math.h>
#include "SIMD.h"
#include "types.h"

namespace Ruined
{
	namespace Math
	{
	#pragma region FLOAT2
		struct Vector2f : public float2
		{		
		#ifdef __SIMD__
			Vector2f (SIMD::vFloat v);
		#endif
			Vector2f(void) {}
			Vector2f(float const value);
			Vector2f(float const x, float const y);

			bool operator==(const Vector2f & other) const;
			bool operator!=(const Vector2f & other) const;

			Vector2f & operator= (const Vector2f & other);

			Vector2f operator+ () const;
			Vector2f operator- () const;

			Vector2f & operator+= (const Vector2f & other);
			Vector2f & operator-= (const Vector2f & other);
			Vector2f & operator*= (const Vector2f & other);
			Vector2f & operator/= (const Vector2f & other);

			Vector2f & operator*= (const float value);
			Vector2f & operator/= (const float value);


				// Resticts a value to be within a specific range.
			static Vector2f	Clamp(const Vector2f & value1, const Vector2f & min, const Vector2f & max);
				// Resticts a value to be within a specific range.
			static void		Clamp(const Vector2f & value1, const Vector2f & min, const Vector2f & max, Vector2f & result);

				// Calculates the distance between two Vector2f's.
			static float	Distance(const Vector2f & value1, const Vector2f & value2);
				// Returns the distance between two Vector2f's.
			static void		Distance(const Vector2f & value1, const Vector2f & value2, float & result);
				// Calculates the distance squared between two Vector2f's.
			static float	DistanceSquared(const Vector2f & value1, const Vector2f & value2);
				// Returns the distance squared between two Vector2f's.
			static void		DistanceSquared(const Vector2f & value1, const Vector2f & value2, float & result);
				
				// Calculates the dot product of two Vector2f's.
			static float	Dot(const Vector2f & value1, const Vector2f & value2);
				// Returns the dot product of two Vector2f's.
			static void		Dot(const Vector2f & value1, const Vector2f & value2, float & result);

				// Caclulates the length of the Vector2f.
			float			Length(void) const;
				// Caclulates the length of the Vector2f squared.
			float			LengthSquared(void) const;

				// Calculates the linear interpolation between two Vector2f's
			static Vector2f	Lerp(const Vector2f & value1, const Vector2f & value2, float const amount);
				// Returns the linear interpolation between two Vector2f's
			static void		Lerp(const Vector2f & value1, const Vector2f & value2, float const amount, Vector2f & result);

				// Returns a vector that contains the highest value from each matching pair of components.
			static Vector2f	Max(const Vector2f & value1, const Vector2f & value2);
				// Returns a vector that contains the lowest value from each matching pair of components.
			static Vector2f	Min(const Vector2f & value1, const Vector2f & value2);	

				// Normalizes the current Vector2f.
			void			Normalize(void);
				// Creates a normalized version of the specified Vector2f.
			static Vector2f	Normalize (const Vector2f & value);
				// Returns a normalized version of the specified Vector2f.
			static void		Normalize (const Vector2f & value, Vector2f & result);
		#ifdef __SIMD__
				// Approximates the normalized version of the current Vector2f.
			void			FastNormalize(void);
				// Creates an approximate normalized version of the specified Vector2f.
			static Vector2f	FastNormalize (const Vector2f & value);
				// Returns an approximate normalized version of the specified Vector2f.
			static void		FastNormalize (const Vector2f & value, Vector2f & result);
		#endif

				// Returns a Vector2f with all of its members set to 1.0f :
				//	returns (1.0f, 1.0f)
			static Vector2f	One(void);
				// Returns a Vector2f with the x member set to 1.0f :
				//	returns (1.0f, 0.0f)
			static Vector2f	UnitX(void);
				// Returns a Vector2f with the y member set to 1.0f :
				//	returns (0.0f, 1.0f)
			static Vector2f	UnitY(void);
				// Returns a Vector2f with all of its members set to 0.0f :
				//	returns (0.0f, 0.0f)
			static Vector2f	Zero(void);
		};

		inline Vector2f operator+ (const Vector2f & f1, const Vector2f & f2);
		inline Vector2f operator- (const Vector2f & f1, const Vector2f & f2);
		inline Vector2f operator* (const Vector2f & f1, const Vector2f & f2);
		inline Vector2f operator/ (const Vector2f & f1, const Vector2f & f2);
		
		inline Vector2f operator* (const Vector2f & fv, const float value);
		inline Vector2f operator* (const float value, const Vector2f & fv);
		inline Vector2f operator/ (const Vector2f & fv, const float value);
		inline Vector2f operator/ (const float value, const Vector2f & fv);
	#pragma endregion
	#pragma region FLOAT3
		struct Vector3f : public float3
		{
		#ifdef __SIMD__
			Vector3f (SIMD::vFloat v);
		#endif
			Vector3f(void) {}
			Vector3f(float const value);
			Vector3f(float const x, float const y, float const z);

			bool operator==(const Vector3f & other) const;
			bool operator!=(const Vector3f & other) const;

			Vector3f & operator= (const Vector3f & other);

			Vector3f operator+ () const;
			Vector3f operator- () const;

			Vector3f & operator+= (const Vector3f & other);
			Vector3f & operator-= (const Vector3f & other);
			Vector3f & operator*= (const Vector3f & other);
			Vector3f & operator/= (const Vector3f & other);

			Vector3f & operator*= (const float value);
			Vector3f & operator/= (const float value);


				// Calculates the cross product of two Vector3f's.
			static Vector3f	Cross(const Vector3f & value1, const Vector3f & value2);
				// Returns the cross product of two Vector3f's.
			static void		Cross(const Vector3f & value1, const Vector3f & value2, Vector3f & result);

				// Resticts a value to be within a specific range.
			static Vector3f	Clamp(const Vector3f & value1, const Vector3f & min, const Vector3f & max);
				// Resticts a value to be within a specific range.
			static void		Clamp(const Vector3f & value1, const Vector3f & min, const Vector3f & max, Vector3f & result);

				// Calculates the distance between two Vector3f's.
			static float	Distance(const Vector3f & value1, const Vector3f & value2);
				// Returns the distance between two Vector3f's.
			static void		Distance(const Vector3f & value1, const Vector3f & value2, float & result);
				// Calculates the distance squared between two Vector3f's.
			static float	DistanceSquared(const Vector3f & value1, const Vector3f & value2);
				// Returns the distance squared between two Vector3f's.
			static void		DistanceSquared(const Vector3f & value1, const Vector3f & value2, float & result);

				// Calculates the dot product of two Vector3f's.
			static float	Dot(const Vector3f & value1, const Vector3f & value2);
				// Returns the dot product of two Vector3f's.
			static void		Dot(const Vector3f & value1, const Vector3f & value2, float & result);
				

				// Caclulates the length of the Vector3f.
			float			Length(void) const;
				// Caclulates the length of the Vector3f squared.
			float			LengthSquared(void) const;

				// Calculates the linear interpolation between two Vector3f's
			static Vector3f	Lerp(const Vector3f & value1, const Vector3f & value2, float const amount);
				// Returns the linear interpolation between two Vector3f's
			static void		Lerp(const Vector3f & value1, const Vector3f & value2, float const amount, Vector3f & result);

				// Returns a vector that contains the highest value from each matching pair of components.
			static Vector3f	Max(const Vector3f & value1, const Vector3f & value2);
				// Returns a vector that contains the lowest value from each matching pair of components.
			static Vector3f	Min(const Vector3f & value1, const Vector3f & value2);	

				// Normalizes the current Vector3f.
			void			Normalize(void);
				// Creates a normalized version of the specified Vector3f.
			static Vector3f	Normalize (const Vector3f & value);
				// Returns a normalized version of the specified Vector3f.
			static void		Normalize (const Vector3f & value, Vector3f & result);
		#ifdef __SIMD__
				// Approximates the normalized version of the current Vector3f.
			void			FastNormalize(void);
				// Creates an approximate normalized version of the specified Vector3f.
			static Vector3f	FastNormalize (const Vector3f & value);
				// Returns an approximate normalized version of the specified Vector3f.
			static void		FastNormalize (const Vector3f & value, Vector3f & result);
		#endif
				// Returns a Vector3f unit vector designating backward in the right-handed coordinate system :
				//	returns (0.0f, 0.0f, 1.0f)
			static Vector3f	Backward(void);
				// Returns a Vector3f unit vector designating down :
				//	returns (0.0f, -1.0f, 0.0f)
			static Vector3f	Down(void);
				// Returns a Vector3f unit vector designating forward in the right-handed coordinate system :
				//	returns (0.0f, 0.0f, -1.0f)
			static Vector3f	Forward(void);
				// Returns a Vector3f unit vector designating left :
				//	returns (-1.0f, 0.0f, 0.0f)
			static Vector3f	Left(void);
				// Returns a Vector3f with all of its members set to 1.0f :
				//	returns (1.0f, 1.0f, 1.0f)
			static Vector3f	One(void);
				// Returns a Vector3f unit vector designating right :
				//	returns (1.0f, 0.0f, 0.0f)
			static Vector3f	Right(void);
				// Returns a Vector3f with the x member set to 1.0f :
				//	returns (1.0f, 0.0f, 0.0f)
			static Vector3f	UnitX(void);
				// Returns a Vector3f with the y member set to 1.0f :
				//	returns (0.0f, 1.0f, 0.0f)
			static Vector3f	UnitY(void);
				// Returns a Vector3f with the z member set to 1.0f :
				//	returns (0.0f, 0.0f, 1.0f)
			static Vector3f	UnitZ(void);
				// Returns a Vector3f unit vector designating up :
				//	returns (0.0f, 1.0f, 0.0f)
			static Vector3f	Up(void);
				// Returns a Vector3f with all of its members set to 0.0f :
				//	returns (0.0f, 0.0f, 0.0f)
			static Vector3f	Zero(void);

		};

		inline Vector3f operator+ (const Vector3f & f1, const Vector3f & f2);
		inline Vector3f operator- (const Vector3f & f1, const Vector3f & f2);
		inline Vector3f operator* (const Vector3f & f1, const Vector3f & f2);
		inline Vector3f operator/ (const Vector3f & f1, const Vector3f & f2);
		
		inline Vector3f operator* (const Vector3f & fv, const float value);
		inline Vector3f operator* (const float value, const Vector3f & fv);
		inline Vector3f operator/ (const Vector3f & fv, const float value);
		inline Vector3f operator/ (const float value, const Vector3f & fv);
	#pragma endregion
	#pragma region FLOAT4
		struct ALIGN(16) Vector4f : public float4
		{
		#ifdef __SIMD__
			Vector4f (SIMD::vFloat v) { SIMD::Store(&x, v); }
		#endif
			Vector4f (void) {}
			Vector4f (float const value);	
			Vector4f (float const x, float const y, float const z, float const w);

			bool		operator== (const Vector4f & other) const;
			bool		operator!= (const Vector4f & other) const;

			Vector4f &	operator= (const float4 & other);

			Vector4f		operator+ () const;
			Vector4f		operator- () const;

			Vector4f &	operator+= (const Vector4f & other);
			Vector4f &	operator-= (const Vector4f & other);
			Vector4f &	operator*= (const Vector4f & other);
			Vector4f &	operator/= (const Vector4f & other);

			Vector4f &	operator*= (const float value);
			Vector4f &	operator/= (const float value);

				// Resticts a value to be within a specific range.
			static Vector4f	Clamp(const Vector4f & value1, const Vector4f & min, const Vector4f & max);
				// Resticts a value to be within a specific range.
			static void		Clamp(const Vector4f & value1, const Vector4f & min, const Vector4f & max, Vector4f & result);

				// Calculates the distance between two Vector4f's.
			static float	Distance(const Vector4f & value1, const Vector4f & value2);
				// Returns the distance between two Vector4f's.
			static void		Distance(const Vector4f & value1, const Vector4f & value2, float & result);
				// Calculates the distance squared between two Vector4f's.
			static float	DistanceSquared(const Vector4f & value1, const Vector4f & value2);
				// Returns the distance squared between two Vector4f's.
			static void		DistanceSquared(const Vector4f & value1, const Vector4f & value2, float & result);

				// Calculates the dot product of two Vector4f's.
			static float	Dot(const Vector4f & value1, const Vector4f & value2);
				// Returns the dot product of two Vector4f's.
			static void		Dot(const Vector4f & value1, const Vector4f & value2, float & result);

				// Caclulates the length of the Vector4f.
			float			Length(void) const;
				// Caclulates the length of the Vector4f squared.
			float			LengthSquared(void) const;

				// Calculates the linear interpolation between two Vector4f's
			static Vector4f	Lerp(const Vector4f & value1, const Vector4f & value2, float const amount);
				// Returns the linear interpolation between two Vector4f's
			static void		Lerp(const Vector4f & value1, const Vector4f & value2, float const amount, Vector4f & result);

				// Returns a vector that contains the highest value from each matching pair of components.
			static Vector4f	Max(const Vector4f & value1, const Vector4f & value2);
				// Returns a vector that contains the lowest value from each matching pair of components.
			static Vector4f	Min(const Vector4f & value1, const Vector4f & value2);	
	
				// Normalizes the current Vector4f.
			void			Normalize(void);
				// Creates a normalized version of the specified Vector4f.
			static Vector4f	Normalize (const Vector4f & value);
				// Returns a normalized version of the specified Vector4f.
			static void		Normalize (const Vector4f & value, Vector4f & result);
		#ifdef __SIMD__
				// Approximates the normalized version of the current Vector4f.
			void			FastNormalize(void);
				// Returns an approximate normalized version of the current Vector4f.
			void			FastNormalize(Vector4f & result) const;
				// Creates an approximate normalized version of the specified Vector4f.
			static Vector4f	FastNormalize (const Vector4f & value);
				// Returns an approximate normalized version of the specified Vector4f.
			static void		FastNormalize (const Vector4f & value, Vector4f & result);
		#endif

				// Returns a Vector4f with all of its members set to 1.0f :
				//	returns (1.0f, 1.0f, 1.0f, 1.0f)
			static Vector4f	One(void);
				// Returns a Vector4f with the x member set to 1.0f :
				//	returns (1.0f, 0.0f, 0.0f, 0.0f)
			static Vector4f	UnitX(void);
				// Returns a Vector4f with the y member set to 1.0f :
				//	returns (0.0f, 1.0f, 0.0f, 0.0f)
			static Vector4f	UnitY(void);
				// Returns a Vector4f with the z member set to 1.0f :
				//	returns (0.0f, 0.0f, 1.0f, 0.0f)
			static Vector4f	UnitZ(void);
				// Returns a Vector4f with the w member set to 1.0f :
				//	returns (0.0f, 0.0f, 0.0f, 1.0f)
			static Vector4f	UnitW(void);
				// Returns a Vector4f with all of its members set to 0.0f :
				//	returns (0.0f, 0.0f, 0.0f, 0.0f)
			static Vector4f	Zero(void);
		};

		Vector4f operator+ (const Vector4f & f1, const Vector4f & f2);
		Vector4f operator- (const Vector4f & f1, const Vector4f & f2);
		Vector4f operator* (const Vector4f & f1, const Vector4f & f2);
		Vector4f operator/ (const Vector4f & f1, const Vector4f & f2);
		
		Vector4f operator* (const Vector4f & f, const float value);
		Vector4f operator* (const float value, const Vector4f & f);
		Vector4f operator/ (const Vector4f & f, const float value);
		Vector4f operator/ (const float value, const Vector4f & f);	
	#pragma endregion

		typedef Vector2f Vector2;
		typedef Vector3f Vector3;
		typedef Vector4f Vector4;
	}
}

#include "vector_f.hpp"