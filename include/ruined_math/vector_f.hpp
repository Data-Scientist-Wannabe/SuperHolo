#include <assert.h>

namespace Ruined
{
	namespace Math
	{
	#pragma region FLOAT2
	#ifdef __SIMD__
			// Loads a Vector2f into a vFloat
			//	returns (f.x, f.y, 0, 0)
		inline SIMD::vFloat LoadFloat2 (const Vector2f & f)
		{
			SIMD::vFloat x = SIMD::LoadS( &f.x );
			SIMD::vFloat y = SIMD::LoadS( &f.y );
			return SIMD::UnpackLow( x, y );
		}
			// Stores the lower three floating-point values
		inline void			StoreFloat2(Vector2f & f, SIMD::vFloat v)
		{
			SIMD::StoreS( &f.x, v );
			SIMD::vFloat vT = SIMD_REORDER_PS(v, _MM_SHUFFLE(1,1,1,1));
			SIMD::StoreS( &f.y, vT );
		}
		inline				Vector2f::Vector2f (SIMD::vFloat v)
		{
			StoreFloat2(*this, v);
		}
	#endif

		inline			Vector2f::Vector2f(float const value)
								{ x = y = value; }
		inline			Vector2f::Vector2f(float const x, float const y)
								{ this->x = x; this->y = y; }

		inline bool		Vector2f::operator==(const Vector2f & other) const 
		{	
			return (other.x == x && other.y == y);
		}
		inline bool		Vector2f::operator!=(const Vector2f & other) const 
		{
			return !(*this == other);
		}

		inline Vector2f &	Vector2f::operator= (const Vector2f & other)
		{
			if(this == &other)
				return *this;

			x = other.x;
			y = other.y;

			return *this;
		}

		inline Vector2f Vector2f::operator+ () const { return *this; }
		inline Vector2f Vector2f::operator- () const { return Vector2f(-x, -y); }

		inline Vector2f &	Vector2f::operator+= (const Vector2f & other)
		{
			#ifdef __SIMD__
				SIMD::vFloat v1 = LoadFloat2(*this);
				SIMD::vFloat v2 = LoadFloat2(other);
				v1 = SIMD::Add(v1, v2);
				StoreFloat2(*this, v1);
			#else
				x += other.x;
				y += other.y;
			#endif
			return *this;
		}
		inline Vector2f & Vector2f::operator-= (const Vector2f & other)
		{
			#ifdef __SIMD__
				SIMD::vFloat v1 = LoadFloat2(*this);
				SIMD::vFloat v2 = LoadFloat2(other);
				v1 = SIMD::Subtract(v1, v2);
				StoreFloat2(*this, v1);
			#else
				x -= other.x;
				y -= other.y;
			#endif
			return *this;
		}
		inline Vector2f &	Vector2f::operator*= (const Vector2f & other)
		{
			#ifdef __SIMD__
				SIMD::vFloat v1 = LoadFloat2(*this);
				SIMD::vFloat v2 = LoadFloat2(other);
				v1 = SIMD::Multiply(v1, v2);
				StoreFloat2(*this, v1);
			#else
				x *= other.x;
				y *= other.y;
			#endif
			return *this;
		}
		inline Vector2f &	Vector2f::operator/= (const Vector2f & other)
		{
			#ifdef __SIMD__
				SIMD::vFloat v1 = LoadFloat2(*this);
				SIMD::vFloat v2 = LoadFloat2(other);
				v1 = SIMD::Divide(v1, v2);
				StoreFloat2(*this, v1);
			#else
				x /= other.x;
				y /= other.y;
			#endif
			return *this;
		}

		inline Vector2f &	Vector2f::operator*= (const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat2(*this);
			SIMD::vFloat s = SIMD::Set1(value);
			v = SIMD::Multiply(v, s);
			StoreFloat2(*this, v);
		#else
			x *= value;
			y *= value;
		#endif
			return *this;
		}
		inline Vector2f &	Vector2f::operator/= (const float value)
		{
			assert(value != 0.0f);
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat2(*this);
			SIMD::vFloat s = SIMD::Set1(value);
			v = SIMD::Divide(v, s);
			StoreFloat2(*this, v);
		#else
			x /= value;
			y /= value;
		#endif
			return *this;
		}

		inline Vector2f	Vector2f::Clamp(const Vector2f & value1, const Vector2f & min, const Vector2f & max)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat vM = LoadFloat2(min);
			v1 = SIMD::Max(v1, vM);
			vM = LoadFloat2(max);
			v1 = SIMD::Min(v1, vM);
			return Vector2f(v1);
		#else
			return Vector2f
				(
					value1.x > max.x ? max.x : (value1.x < min.x ? min.x : value1.x),
					value1.y > max.y ? max.y : (value1.y < min.x ? min.y : value1.y)
				);
		#endif
		}
		inline void		Vector2f::Clamp(const Vector2f & value1, const Vector2f & min, const Vector2f & max, Vector2f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat vM = LoadFloat2(min);
			v1 = SIMD::Max(v1, vM);
			vM = LoadFloat2(max);
			v1 = SIMD::Min(v1, vM);
			StoreFloat2(result, v1);
		#else
			result = Vector2f
				(
					value1.x > max.x ? max.x : (value1.x < min.x ? min.x : value1.x),
					value1.y > max.y ? max.y : (value1.y < min.x ? min.y : value1.y)
				);
		#endif
		}

		inline float	Vector2f::Distance(const Vector2f & value1, const Vector2f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::SqrtS(v1);
			return SIMD::GetX(v1);
		#else
			return (value1 - value2).Length();
		#endif
		}
		inline void		Vector2f::Distance(const Vector2f & value1, const Vector2f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::SqrtS(v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1 - value2).Length();
		#endif
		}
		inline float	Vector2f::DistanceSquared(const Vector2f & value1, const Vector2f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			return SIMD::GetX(v1);
		#else
			return (value1 - value2).LengthSquared();
		#endif
		}
		inline void		Vector2f::DistanceSquared(const Vector2f & value1, const Vector2f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1 - value2).LengthSquared();
		#endif
		}

		inline float	Vector2f::Dot(const Vector2f & value1, const Vector2f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			v1 = SIMD::Multiply(v1, v2);
			v1 = SIMD::AddH(v1, v1);
			return SIMD::GetX(v1);
		#else
			return (value1.x * value2.x) + (value1.y * value2.y);
		#endif
		}
		inline void		Vector2f::Dot(const Vector2f & value1, const Vector2f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			v1 = SIMD::Multiply(v1, v2);
			v1 = SIMD::AddH(v1, v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1.x * value2.x) + (value1.y * value2.y);
		#endif
		}

		inline float	Vector2f::Length(void) const
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat2(*this);
			v = SIMD::Multiply(v, v);
			v = SIMD::AddH(v, v);
			v = SIMD::SqrtS(v);
			return SIMD::GetX(v);
		#else
			return sqrtf(x*x + y*y);
		#endif
				
		}
		inline float	Vector2f::LengthSquared(void) const
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat2(*this);
			SIMD::Multiply(v, v);
			v = SIMD::AddH(v, v);
			return SIMD::GetX(v);
		#else
			return x*x + y*y;
		#endif			
		}

		inline Vector2f	Vector2f::Lerp(const Vector2f & value1, const Vector2f & value2, float const amount)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			SIMD::vFloat vA = SIMD::Set1(amount);
			v2 = SIMD::Subtract(v2, v1);
			v2 = SIMD::Multiply(v2, vA);
			v2 = SIMD::Add(v2, v1);
			return Vector2f(v2);
		#else
			return Vector2f
				(
					value1.x + (value2.x - value1.x) * amount,
					value1.y + (value2.y - value1.y) * amount
				);
		#endif
		}
		inline void		Vector2f::Lerp(const Vector2f & value1, const Vector2f & value2, float const amount, Vector2f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			SIMD::vFloat vA = SIMD::Set1(amount);
			v2 = SIMD::Subtract(v2, v1);
			v2 = SIMD::Multiply(v2, vA);
			v2 = SIMD::Add(v2, v1);
			StoreFloat2(result, v2);
		#else
			result = Vector2f
				(
					value1.x + (value2.x - value1.x) * amount,
					value1.y + (value2.y - value1.y) * amount
				);
		#endif

		}

		inline Vector2f	Vector2f::Max(const Vector2f & value1, const Vector2f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			v1 = SIMD::Max(v1, v2);
			return Vector2f(v1);
		#else
			return Vector2f
				(
					value1.x > value2.x ? value1.x : value2.x,
					value1.y > value2.y ? value1.y : value2.y
				);
		#endif
		}
		inline Vector2f	Vector2f::Min(const Vector2f & value1, const Vector2f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(value1);
			SIMD::vFloat v2 = LoadFloat2(value2);
			v1 = SIMD::Min(v1, v2);
			return Vector2f(v1);
		#else
			return Vector2f
				(
					value1.x < value2.x ? value1.x : value2.x,
					value1.y < value2.y ? value1.y : value2.y
				);
		#endif
		}

		inline void		Vector2f::Normalize(void)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat2(*this);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			v = SIMD::Divide(v, s);
			StoreFloat2(*this, v);
		#else
			float mag = Length();
			x /= mag;
			y /= mag;
		#endif
		}
		inline Vector2f	Vector2f::Normalize(const Vector2f & value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat2(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			v = SIMD::Divide(v, s);
			return Vector2f(v);
		#else
			float mag = value.Length();
			return Vector2f(value.x / mag, value.y / mag);
		#endif
		}
		inline void		Vector2f::Normalize(const Vector2f & value, Vector2f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat2(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			v = SIMD::Divide(v, s);
			StoreFloat2(result, v);
		#else
			float mag = value.Length();
			result.x = value.x / mag;
			result.y = value.y / mag;
		#endif
		}
	#ifdef __SIMD__
		inline void		Vector2f::FastNormalize(void)
		{
			SIMD::vFloat v = LoadFloat2(*this);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			StoreFloat2(*this, v);
		}
		inline Vector2f	Vector2f::FastNormalize(const Vector2f & value)
		{
			SIMD::vFloat v = LoadFloat2(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			return Vector2f(v);
		}
		inline void		Vector2f::FastNormalize(const Vector2f & value, Vector2f & result)
		{
			SIMD::vFloat v = LoadFloat2(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			StoreFloat2(result, v);
		}
	#endif

		inline Vector2f	Vector2f::One(void)		{ return Vector2f( 1.0f,  1.0f); }
		inline Vector2f	Vector2f::UnitX(void)		{ return Vector2f( 1.0f,  0.0f); }		
		inline Vector2f	Vector2f::UnitY(void)		{ return Vector2f( 0.0f,  1.0f); }
		inline Vector2f	Vector2f::Zero(void)		{ return Vector2f( 0.0f,  0.0f); }  

		inline Vector2f operator+ (const Vector2f & f1, const Vector2f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(f1);
			SIMD::vFloat v2 = LoadFloat2(f2);
			v1 = SIMD::Add(v1, v2);
			return Vector2f(v1);
		#else
			return Vector2f(f1.x + f2.x, f1.y + f2.y);
		#endif
		}
		inline Vector2f operator- (const Vector2f & f1, const Vector2f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(f1);
			SIMD::vFloat v2 = LoadFloat2(f2);
			v1 = SIMD::Subtract(v1, v2);
			return Vector2f(v1);
		#else
			return Vector2f(f1.x - f2.x, f1.y - f2.y);
		#endif
		}
		inline Vector2f operator* (const Vector2f & f1, const Vector2f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(f1);
			SIMD::vFloat v2 = LoadFloat2(f2);
			v1 = SIMD::Multiply(v1, v2);
			return Vector2f(v1);
		#else
			return Vector2f(f1.x * f2.x, f1.y * f2.y);
		#endif
		}
		inline Vector2f operator/ (const Vector2f & f1, const Vector2f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(f1);
			SIMD::vFloat v2 = LoadFloat2(f2);
			v1 = SIMD::Divide(v1, v2);
			return Vector2f(v1);
		#else
			return Vector2f(f1.x / f2.x, f1.y / f2.y);
		#endif
		}
		
		inline Vector2f operator* (const Vector2f & f, const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(f);
			SIMD::vFloat v2 = SIMD::Set1(value);
			v1 = SIMD::Multiply(v1, v2);
			return Vector2f(v1);
		#else
			return Vector2f(f.x * value, f.y * value);
		#endif
		}
		inline Vector2f operator* (const float value, const Vector2f & f)
		{
			return operator*(f, value);
		}
		inline Vector2f operator/ (const Vector2f & f, const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat2(f);
			SIMD::vFloat v2 = SIMD::Set1(value);
			v1 = SIMD::Divide(v1, v2);
			return Vector2f(v1);
		#else
			return Vector2f(f.x / value, f.y / value);
		#endif
		}
		inline Vector2f operator/ (const float value, const Vector2f & f)
		{
			return operator/(f, value);
		}
	#pragma endregion
	#pragma region FLOAT3
	#ifdef __SIMD__
			// Loads a Vector3f into a vFloat
			//	returns (f.x, f.y, f.z, 0)
		inline SIMD::vFloat LoadFloat3 (const Vector3f & f)
		{
			SIMD::vFloat x = SIMD::LoadS( &f.x );
			SIMD::vFloat y = SIMD::LoadS( &f.y );
			SIMD::vFloat z = SIMD::LoadS( &f.z );
			SIMD::vFloat xy = SIMD::UnpackLow( x, y );
			return SIMD::MoveLowHalf(xy, z);
		}
			// Stores the lower three floating-point values
		inline void			StoreFloat3(Vector3f & f, SIMD::vFloat v)
		{
			SIMD::StoreS( &f.x, v );
			SIMD::vFloat vT = SIMD_REORDER_PS(v, _MM_SHUFFLE(1,1,1,1));
			SIMD::StoreS( &f.y, vT );
			vT = SIMD_REORDER_PS(v, _MM_SHUFFLE(2,2,2,2));
			SIMD::StoreS( &f.z, vT );
		}
		inline				Vector3f::Vector3f (SIMD::vFloat v)
		{
			StoreFloat3(*this, v);
		}
	#endif

		inline			Vector3f::Vector3f(float const value)
								{ x = y = z = value; }
		inline			Vector3f::Vector3f(float const x, float const y, float const z)
								{ this->x = x; this->y = y; this->z = z; }

		inline bool		Vector3f::operator==(const Vector3f & other) const 
		{	
			return (other.x == x && other.y == y) && (other.z == z);
		}
		inline bool		Vector3f::operator!=(const Vector3f & other) const 
		{
			return !(*this == other);
		}

		inline Vector3f &	Vector3f::operator= (const Vector3f & other)
		{
			if (this == &other) {
				return *this;
			}

			x = other.x;
			y = other.y;
			z = other.z;

			return *this;
		}

		inline Vector3f Vector3f::operator+ () const { return *this; }
		inline Vector3f Vector3f::operator- () const { return Vector3f(-x, -y, -z); }

		inline Vector3f &	Vector3f::operator+= (const Vector3f & other)
		{
			#ifdef __SIMD__
				SIMD::vFloat v1 = LoadFloat3(*this);
				SIMD::vFloat v2 = LoadFloat3(other);
				v1 = SIMD::Add(v1, v2);
				StoreFloat3(*this, v1);
			#else
				x += other.x;
				y += other.y;
				z += other.z;
			#endif
			return *this;
		}
		inline Vector3f &	Vector3f::operator-= (const Vector3f & other)
		{
			#ifdef __SIMD__
				SIMD::vFloat v1 = LoadFloat3(*this);
				SIMD::vFloat v2 = LoadFloat3(other);
				v1 = SIMD::Subtract(v1, v2);
				StoreFloat3(*this, v1);
			#else
				x -= other.x;
				y -= other.y;
				z -= other.z;
			#endif
			return *this;
		}
		inline Vector3f &	Vector3f::operator*= (const Vector3f & other)
		{
			#ifdef __SIMD__
				SIMD::vFloat v1 = LoadFloat3(*this);
				SIMD::vFloat v2 = LoadFloat3(other);
				v1 = SIMD::Multiply(v1, v2);
				StoreFloat3(*this, v1);
			#else
				x *= other.x;
				y *= other.y;
				z *= other.z;
			#endif
			return *this;
		}
		inline Vector3f &	Vector3f::operator/= (const Vector3f & other)
		{
			#ifdef __SIMD__
				SIMD::vFloat v1 = LoadFloat3(*this);
				SIMD::vFloat v2 = LoadFloat3(other);
				v1 = SIMD::Divide(v1, v2);
				StoreFloat3(*this, v1);
			#else
				x /= other.x;
				y /= other.y;
				z /= other.z;
			#endif
			return *this;
		}

		inline Vector3f &	Vector3f::operator*= (const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat3(*this);
			SIMD::vFloat s = SIMD::Set1(value);
			v = SIMD::Multiply(v, s);
			StoreFloat3(*this, v);
		#else
			x *= value;
			y *= value;
			z *= value;
		#endif
			return *this;
		}
		inline Vector3f &	Vector3f::operator/= (const float value)
		{
			assert(value != 0.0f);
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat3(*this);
			SIMD::vFloat s = SIMD::Set1(value);
			v = SIMD::Divide(v, s);
			StoreFloat3(*this, v);
		#else
			x /= value;
			y /= value;
			z /= value;
		#endif
			return *this;
		}

		inline Vector3f	Vector3f::Cross(const Vector3f & value1, const Vector3f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1); // A
			SIMD::vFloat v2 = LoadFloat3(value2); // B

			SIMD::vFloat vT1 = SIMD_REORDER_PS(v1, _MM_SHUFFLE(3, 0, 2, 1)); // result (A.y, A.z, A.x, A.w)
			SIMD::vFloat vT2 = SIMD_REORDER_PS(v2, _MM_SHUFFLE(3, 1, 0, 2)); // result (B.z, B.x, B.y, B.w)

			SIMD::vFloat vR = SIMD::Multiply(vT1, vT2);

			vT1 = SIMD_REORDER_PS(vT1, _MM_SHUFFLE(3, 0, 2, 1)); // result (A.z, A.x, A.y, A.w)
			vT2 = SIMD_REORDER_PS(vT2, _MM_SHUFFLE(3, 1, 0, 2)); // result (B.y, B.z, B.x, B.w)

			vT1 = SIMD::Multiply(vT1, vT2);

			return Vector3f(SIMD::Subtract(vR, vT1));  
		#else
			return Vector3f
				(	( value1.y * value2.z ) - ( value1.z * value2.y ),
					( value1.z * value2.x ) - ( value1.x * value2.z ),
					( value1.x * value2.y ) - ( value1.y * value2.x )	);
		#endif
		}
		inline void		Vector3f::Cross(const Vector3f & value1, const Vector3f & value2, Vector3f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1); // A
			SIMD::vFloat v2 = LoadFloat3(value2); // B

			SIMD::vFloat vT1 = SIMD_REORDER_PS(v1, _MM_SHUFFLE(3, 0, 2, 1)); // result (A.y, A.z, A.x, A.w)
			SIMD::vFloat vT2 = SIMD_REORDER_PS(v2, _MM_SHUFFLE(3, 1, 0, 2)); // result (B.z, B.x, B.y, B.w)

			SIMD::vFloat vR = SIMD::Multiply(vT1, vT2);

			vT1 = SIMD_REORDER_PS(vT1, _MM_SHUFFLE(3, 0, 2, 1)); // result (A.z, A.x, A.y, A.w)
			vT2 = SIMD_REORDER_PS(vT2, _MM_SHUFFLE(3, 1, 0, 2)); // result (B.y, B.z, B.x, B.w)

			vT1 = SIMD::Multiply(vT1, vT2);

			StoreFloat3(result, SIMD::Subtract(vR, vT1));  
		#else
			result = Vector3f
				(	( value1.y * value2.z ) - ( value1.z * value2.y ),
					( value1.z * value2.x ) - ( value1.x * value2.z ),
					( value1.x * value2.y ) - ( value1.y * value2.x )	);
		#endif
		}

		inline Vector3f	Vector3f::Clamp(const Vector3f & value1, const Vector3f & min, const Vector3f & max)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat vM = LoadFloat3(min);
			v1 = SIMD::Max(v1, vM);
			vM = LoadFloat3(max);
			v1 = SIMD::Min(v1, vM);
			return Vector3f(v1);
		#else
			return Vector3f
				(
					value1.x > max.x ? max.x : (value1.x < min.x ? min.x : value1.x),
					value1.y > max.y ? max.y : (value1.y < min.x ? min.y : value1.y),
					value1.z > max.z ? max.z : (value1.z < min.z ? min.x : value1.z)
				);
		#endif
		}
		inline void		Vector3f::Clamp(const Vector3f & value1, const Vector3f & min, const Vector3f & max, Vector3f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat vM = LoadFloat3(min);
			v1 = SIMD::Max(v1, vM);
			vM = LoadFloat3(max);
			v1 = SIMD::Min(v1, vM);
			StoreFloat3(result, v1);
		#else
			result = Vector3f
				(
					value1.x > max.x ? max.x : (value1.x < min.x ? min.x : value1.x),
					value1.y > max.y ? max.y : (value1.y < min.x ? min.y : value1.y),
					value1.z > max.z ? max.z : (value1.z < min.z ? min.x : value1.z)
				);
		#endif
		}

		inline float	Vector3f::Distance(const Vector3f & value1, const Vector3f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::SqrtS(v1);
			return SIMD::GetX(v1);
		#else
			return (value1 - value2).Length();
		#endif
		}
		inline void		Vector3f::Distance(const Vector3f & value1, const Vector3f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::SqrtS(v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1 - value2).Length();
		#endif
		}
		inline float	Vector3f::DistanceSquared(const Vector3f & value1, const Vector3f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			return SIMD::GetX(v1);
		#else
			return (value1 - value2).LengthSquared();
		#endif
		}
		inline void		Vector3f::DistanceSquared(const Vector3f & value1, const Vector3f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1 - value2).LengthSquared();
		#endif
		}

		inline float	Vector3f::Dot(const Vector3f & value1, const Vector3f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			v1 = SIMD::Multiply(v1, v2);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			return SIMD::GetX(v1);
		#else
			return (value1.x * value2.x) + (value1.y * value2.y) + (value1.z * value2.z);
		#endif
		}
		inline void		Vector3f::Dot(const Vector3f & value1, const Vector3f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			v1 = SIMD::Multiply(v1, v2);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1.x * value2.x) + (value1.y * value2.y) + (value1.z * value2.z);
		#endif
		}

		inline float	Vector3f::Length(void) const
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat3(*this);
			v = SIMD::Multiply(v, v);
			v = SIMD::AddH(v, v);
			v = SIMD::AddH(v, v);
			v = SIMD::SqrtS(v);
			return SIMD::GetX(v);
		#else
			return sqrtf(x*x + y*y + z*z);
		#endif
				
		}
		inline float	Vector3f::LengthSquared(void) const
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat3(*this);
			SIMD::Multiply(v, v);
			v = SIMD::AddH(v, v);
			v = SIMD::AddH(v, v);
			return SIMD::GetX(v);
		#else
			return x*x + y*y + z*z;
		#endif			
		}

		inline Vector3f	Vector3f::Lerp(const Vector3f & value1, const Vector3f & value2, float const amount)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			SIMD::vFloat vA = SIMD::Set1(amount);
			v2 = SIMD::Subtract(v2, v1);
			v2 = SIMD::Multiply(v2, vA);
			v2 = SIMD::Add(v2, v1);
			return Vector3f(v2);
		#else
			return Vector3f
				(
					value1.x + (value2.x - value1.x) * amount,
					value1.y + (value2.y - value1.y) * amount,
					value1.z + (value2.z - value1.z) * amount
				);
		#endif
		}
		inline void		Vector3f::Lerp(const Vector3f & value1, const Vector3f & value2, float const amount, Vector3f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			SIMD::vFloat vA = SIMD::Set1(amount);
			v2 = SIMD::Subtract(v2, v1);
			v2 = SIMD::Multiply(v2, vA);
			v2 = SIMD::Add(v2, v1);
			StoreFloat3(result, v2);
		#else
			result = Vector3f
				(
					value1.x + (value2.x - value1.x) * amount,
					value1.y + (value2.y - value1.y) * amount,
					value1.z + (value2.z - value1.z) * amount
				);
		#endif

		}

		inline Vector3f	Vector3f::Max(const Vector3f & value1, const Vector3f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			v1 = SIMD::Max(v1, v2);
			return Vector3f(v1);
		#else
			return Vector3f
				(
					value1.x > value2.x ? value1.x : value2.x,
					value1.y > value2.y ? value1.y : value2.y,
					value1.z > value2.z ? value1.z : value2.z
				);
		#endif
		}
		inline Vector3f	Vector3f::Min(const Vector3f & value1, const Vector3f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(value1);
			SIMD::vFloat v2 = LoadFloat3(value2);
			v1 = SIMD::Min(v1, v2);
			return Vector3f(v1);
		#else
			return Vector3f
				(
					value1.x < value2.x ? value1.x : value2.x,
					value1.y < value2.y ? value1.y : value2.y,
					value1.z < value2.z ? value1.z : value2.z
				);
		#endif
		}

		inline void		Vector3f::Normalize(void)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat3(*this);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			v = SIMD::Divide(v, s);
			StoreFloat3(*this, v);
		#else
			float mag = Length();
			x /= mag;
			y /= mag;
			z /= mag;
		#endif
		}
		inline Vector3f	Vector3f::Normalize(const Vector3f & value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat3(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			v = SIMD::Divide(v, s);
			return Vector3f(v);
		#else
			float mag = value.Length();
			return Vector3f(value.x / mag, value.y / mag, value.z / mag);
		#endif
		}
		inline void		Vector3f::Normalize(const Vector3f & value, Vector3f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat3(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			v = SIMD::Divide(v, s);
			StoreFloat3(result, v);
		#else
			float mag = value.Length();
			result.x = value.x / mag;
			result.y = value.y / mag;
			result.z = value.z / mag;
		#endif
		}
	#ifdef __SIMD__
		inline void		Vector3f::FastNormalize(void)
		{
			SIMD::vFloat v = LoadFloat3(*this);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			StoreFloat3(*this, v);
		}
		inline Vector3f	Vector3f::FastNormalize(const Vector3f & value)
		{
			SIMD::vFloat v = LoadFloat3(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			return Vector3f(v);
		}
		inline void		Vector3f::FastNormalize(const Vector3f & value, Vector3f & result)
		{
			SIMD::vFloat v = LoadFloat3(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			StoreFloat3(result, v);
		}
	#endif

		inline Vector3f	Vector3f::Backward(void)	{ return Vector3f( 0.0f,  0.0f,  1.0f); }
		inline Vector3f	Vector3f::Down(void)		{ return Vector3f( 0.0f, -1.0f,  0.0f); }
		inline Vector3f	Vector3f::Forward(void)	{ return Vector3f( 0.0f,  0.0f, -1.0f); }
		inline Vector3f	Vector3f::Left(void)		{ return Vector3f(-1.0f,  0.0f,  0.0f); }
		inline Vector3f	Vector3f::One(void)		{ return Vector3f( 1.0f,  1.0f,  1.0f); }
		inline Vector3f	Vector3f::Right(void)		{ return Vector3f( 1.0f,  0.0f,  0.0f); }
		inline Vector3f	Vector3f::UnitX(void)		{ return Vector3f( 1.0f,  0.0f,  0.0f); }
		inline Vector3f	Vector3f::UnitY(void)		{ return Vector3f( 0.0f,  1.0f,  0.0f); }
		inline Vector3f	Vector3f::UnitZ(void)		{ return Vector3f( 0.0f,  0.0f,  1.0f); }
		inline Vector3f	Vector3f::Up(void)		{ return Vector3f( 0.0f,  1.0f,  0.0f); }
		inline Vector3f	Vector3f::Zero(void)		{ return Vector3f( 0.0f,  0.0f,  0.0f); }

		inline Vector3f operator+ (const Vector3f & f1, const Vector3f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(f1);
			SIMD::vFloat v2 = LoadFloat3(f2);
			v1 = SIMD::Add(v1, v2);
			return Vector3f(v1);
		#else
			return Vector3f(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z);
		#endif
		}
		inline Vector3f operator- (const Vector3f & f1, const Vector3f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(f1);
			SIMD::vFloat v2 = LoadFloat3(f2);
			v1 = SIMD::Subtract(v1, v2);
			return Vector3f(v1);
		#else
			return Vector3f(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z);
		#endif
		}
		inline Vector3f operator* (const Vector3f & f1, const Vector3f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(f1);
			SIMD::vFloat v2 = LoadFloat3(f2);
			v1 = SIMD::Multiply(v1, v2);
			return Vector3f(v1);
		#else
			return Vector3f(f1.x * f2.x, f1.y * f2.y, f1.z * f2.z);
		#endif
		}
		inline Vector3f operator/ (const Vector3f & f1, const Vector3f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(f1);
			SIMD::vFloat v2 = LoadFloat3(f2);
			v1 = SIMD::Divide(v1, v2);
			return Vector3f(v1);
		#else
			return Vector3f(f1.x / f2.x, f1.y / f2.y, f1.z / f2.z);
		#endif
		}
		
		inline Vector3f operator* (const Vector3f & f, const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(f);
			SIMD::vFloat v2 = SIMD::Set1(value);
			v1 = SIMD::Multiply(v1, v2);
			return Vector3f(v1);
		#else
			return Vector3f(f.x * value, f.y * value, f.z * value);
		#endif
		}
		inline Vector3f operator* (const float value, const Vector3f & f)
		{
			return operator*(f, value);
		}
		inline Vector3f operator/ (const Vector3f & f, const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat3(f);
			SIMD::vFloat v2 = SIMD::Set1(value);
			v1 = SIMD::Divide(v1, v2);
			return Vector3f(v1);
		#else
			return Vector3f(f.x / value, f.y / value, f.z / value);
		#endif
		}
		inline Vector3f operator/ (const float value, const Vector3f & f)
		{
			return operator/(f, value);
		}
	#pragma endregion
	#pragma region FLOAT4
	#ifdef __SIMD__
			// Loads a Vector4f into a vFloat
			//	returns (f.x, f.y, f.z, f.w)
		inline SIMD::vFloat LoadFloat4 (const Vector4f & f)
		{
			return SIMD::Load(&f.x);
		}
			// Stores four floating-point values.
		inline void			StoreFloat4(Vector4f & f, SIMD::vFloat v)
		{
			SIMD::Store(&f.x, v);
		}
	#endif
		inline			Vector4f::Vector4f(float const value)
							{ x = y = z = w = value; }
		inline			Vector4f::Vector4f(float const x, float const y, float const z, float const w)
							{ this->x = x; this->y = y; this->z = z; this->w = w; }

		inline bool		Vector4f::operator==(const Vector4f & other) const 
		{	
			return (other.x == x && other.y == y) && (other.z == z) && (other.w == w);
		}
		inline bool		Vector4f::operator!=(const Vector4f & other) const 
		{
			return !(*this == other);
		}

		inline Vector4f & Vector4f::operator= (const float4 & other)
		{
			if(this == &other)
				return *this;

			x = other.x;
			y = other.y;
			z = other.z;
			w = other.w;

			return *this;
		}

		inline Vector4f	Vector4f::operator+ () const { return *this; }
		inline Vector4f	Vector4f::operator- () const { return Vector4f(-x, -y, -z, -w); }

		inline Vector4f &	Vector4f::operator+= (const Vector4f & other)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(*this);
			SIMD::vFloat v2 = LoadFloat4(other);;
			v1 = SIMD::Add(v1, v2);
			SIMD::Store(&x, v1);
		#else
			x += other.x;
			y += other.y;
			z += other.z;
			w += other.w;
		#endif
			return *this;
		}
		inline Vector4f & Vector4f::operator-= (const Vector4f & other)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(*this);
			SIMD::vFloat v2 = LoadFloat4(other);;
			v1 = SIMD::Subtract(v1, v2);
			SIMD::Store(&x, v1);
		#else
			x -= other.x;
			y -= other.y;
			z -= other.z;
			w -= other.w;
		#endif
			return *this;
		}
		inline Vector4f & Vector4f::operator*= (const Vector4f & other)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(*this);
			SIMD::vFloat v2 = LoadFloat4(other);;
			v1 = SIMD::Multiply(v1, v2);
			SIMD::Store(&x, v1);
		#else
			x *= other.x;
			y *= other.y;
			z *= other.z;
			w *= other.w;
		#endif
			return *this;
		}
		inline Vector4f & Vector4f::operator/= (const Vector4f & other)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(*this);
			SIMD::vFloat v2 = LoadFloat4(other);;
			v1 = SIMD::Divide(v1, v2);
			SIMD::Store(&x, v1);
		#else
			x /= other.x;
			y /= other.y;
			z /= other.z;
			w /= other.z;
		#endif			
			return *this;
		}

		inline Vector4f & Vector4f::operator*= (const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = SIMD::Load(&x);
			SIMD::vFloat s = SIMD::Set1(value);
			v = SIMD::Multiply(v, s);
			SIMD::Store(&x, v);
		#else
			x *= value;
			y *= value;
			z *= value;
			w *= value;
		#endif
			return *this;
		}
		inline Vector4f & Vector4f::operator/= (const float value)
		{
			assert(value != 0.0f);
		#ifdef __SIMD__
			SIMD::vFloat v = SIMD::Load(&x);
			SIMD::vFloat s = SIMD::Set1(value);
			v = SIMD::Divide(v, s);
			SIMD::Store(&x, v);
		#else
			x /= value;
			y /= value;
			z /= value;
			w /= value;
		#endif
			return *this;
		}

		inline Vector4f	Vector4f::Clamp(const Vector4f & value1, const Vector4f & min, const Vector4f & max)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat vM = LoadFloat4(min);
			v1 = SIMD::Max(v1, vM);
			vM = LoadFloat4(max);
			v1 = SIMD::Min(v1, vM);
			return Vector4f(v1);
		#else
			return Vector4f
				(
					value1.x > max.x ? max.x : (value1.x < min.x ? min.x : value1.x),
					value1.y > max.y ? max.y : (value1.y < min.x ? min.y : value1.y),
					value1.z > max.z ? max.z : (value1.z < min.z ? min.x : value1.z),
					value1.w > max.w ? max.w : (value1.w < min.w ? min.w : value1.w)
				);
		#endif
		}
		inline void		Vector4f::Clamp(const Vector4f & value1, const Vector4f & min, const Vector4f & max, Vector4f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat vM = LoadFloat4(min);
			v1 = SIMD::Max(v1, vM);
			vM = LoadFloat4(max);
			v1 = SIMD::Min(v1, vM);
			StoreFloat4(result, v1);
		#else
			result = Vector4f
				(
					value1.x > max.x ? max.x : (value1.x < min.x ? min.x : value1.x),
					value1.y > max.y ? max.y : (value1.y < min.x ? min.y : value1.y),
					value1.z > max.z ? max.z : (value1.z < min.z ? min.x : value1.z),
					value1.w > max.w ? max.w : (value1.w < min.w ? min.w : value1.w)
				);
		#endif
		}

		inline float	Vector4f::Distance(const Vector4f & value1, const Vector4f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::SqrtS(v1);
			return SIMD::GetX(v1);
		#else
			return (value1 - value2).Length();
		#endif
		}
		inline void		Vector4f::Distance(const Vector4f & value1, const Vector4f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::SqrtS(v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1 - value2).Length();
		#endif
		}
		inline float	Vector4f::DistanceSquared(const Vector4f & value1, const Vector4f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			return SIMD::GetX(v1);
		#else
			return (value1 - value2).LengthSquared();
		#endif
		}
		inline void		Vector4f::DistanceSquared(const Vector4f & value1, const Vector4f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			v1 = SIMD::Subtract(v1, v2);
			v1 = SIMD::Multiply(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1 - value2).LengthSquared();
		#endif
		}

		inline float	Vector4f::Dot(const Vector4f & value1, const Vector4f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			v1 = SIMD::Multiply(v1, v2);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			return SIMD::GetX(v1);
		#else
			return (value1.x * value2.x) + (value1.y * value2.y) + (value1.z * value2.z) + (value1.w * value2.w);
		#endif
		}
		inline void		Vector4f::Dot(const Vector4f & value1, const Vector4f & value2, float & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			v1 = SIMD::Multiply(v1, v2);
			v1 = SIMD::AddH(v1, v1);
			v1 = SIMD::AddH(v1, v1);
			SIMD::StoreS(&result, v1);
		#else
			result = (value1.x * value2.x) + (value1.y * value2.y) + (value1.z * value2.z) + (value1.w * value2.w);
		#endif
		}

		inline float	Vector4f::Length(void) const
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat4(*this);
			v = SIMD::Multiply(v, v);
			v = SIMD::AddH(v, v);
			v = SIMD::AddH(v, v);
			v = SIMD::SqrtS(v);
			return SIMD::GetX(v);
		#else
			return sqrtf(x*x + y*y + z*z + w*w);
		#endif
				
		}
		inline float	Vector4f::LengthSquared(void) const
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat4(*this);
			SIMD::Multiply(v, v);
			v = SIMD::AddH(v, v);
			v = SIMD::AddH(v, v);
			return SIMD::GetX(v);
		#else
			return x*x + y*y + z*z + w*w;
		#endif			
		}

		inline Vector4f	Vector4f::Lerp(const Vector4f & value1, const Vector4f & value2, float const amount)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			SIMD::vFloat vA = SIMD::Set1(amount);
			v2 = SIMD::Subtract(v2, v1);
			v2 = SIMD::Multiply(v2, vA);
			v2 = SIMD::Add(v2, v1);
			return Vector4f(v2);
		#else
			return Vector4f
				(
					value1.x + (value2.x - value1.x) * amount,
					value1.y + (value2.y - value1.y) * amount,
					value1.z + (value2.z - value1.z) * amount,
					value1.w + (value2.w - value1.w) * amount
				);
		#endif
		}
		inline void		Vector4f::Lerp(const Vector4f & value1, const Vector4f & value2, float const amount, Vector4f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			SIMD::vFloat vA = SIMD::Set1(amount);
			v2 = SIMD::Subtract(v2, v1);
			v2 = SIMD::Multiply(v2, vA);
			v2 = SIMD::Add(v2, v1);
			StoreFloat4(result, v2);
		#else
			result = Vector4f
				(
					value1.x + (value2.x - value1.x) * amount,
					value1.y + (value2.y - value1.y) * amount,
					value1.z + (value2.z - value1.z) * amount,
					value1.w + (value2.w - value1.w) * amount
				);
		#endif

		}

		inline Vector4f	Vector4f::Max(const Vector4f & value1, const Vector4f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			v1 = SIMD::Max(v1, v2);
			return Vector4f(v1);
		#else
			return Vector4f
				(
					value1.x > value2.x ? value1.x : value2.x,
					value1.y > value2.y ? value1.y : value2.y,
					value1.z > value2.z ? value1.z : value2.z,
					value1.w > value2.w ? value1.w : value2.w
				);
		#endif
		}
		inline Vector4f	Vector4f::Min(const Vector4f & value1, const Vector4f & value2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(value1);
			SIMD::vFloat v2 = LoadFloat4(value2);
			v1 = SIMD::Min(v1, v2);
			return Vector4f(v1);
		#else
			return Vector4f
				(
					value1.x < value2.x ? value1.x : value2.x,
					value1.y < value2.y ? value1.y : value2.y,
					value1.z < value2.z ? value1.z : value2.z,
					value1.w < value2.w ? value1.w : value2.w
				);
		#endif
		}

		inline void		Vector4f::Normalize(void)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat4(*this);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			s = SIMD::Reciprocate(s);
			v = SIMD::Multiply(v, s);
			StoreFloat4(*this, v);
		#else
			float mag = Length();
			x /= mag;
			y /= mag;
			z /= mag;
			w /= mag;
		#endif
		}
		inline Vector4f	Vector4f::Normalize(const Vector4f & value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat4(value.x);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			s = SIMD::Reciprocate(s);
			v = SIMD::Multiply(v, s);
			return Vector4f(v);
		#else
			float mag = value.Length();
			return Vector4f(value.x / mag, value.y / mag, value.z / mag, value.w / mag);
		#endif
		}
		inline void		Vector4f::Normalize(const Vector4f & value, Vector4f & result)
		{
		#ifdef __SIMD__
			SIMD::vFloat v = LoadFloat4(value.x);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::Sqrt(s);
			s = SIMD::Reciprocate(s);
			v = SIMD::Multiply(v, s);
			StoreFloat4(result, v);
		#else
			float mag = value.Length();
			result.x = value.x / mag;
			result.y = value.y / mag;
			result.z = value.z / mag;
			result.w = value.w / mag;
		#endif
		}
	#ifdef __SIMD__
		inline void		Vector4f::FastNormalize(void)
		{
			SIMD::vFloat v = LoadFloat4(*this);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			StoreFloat4(*this, v);
		}
		inline void		Vector4f::FastNormalize(Vector4f & result) const
		{
			SIMD::vFloat v = LoadFloat4(*this);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			StoreFloat4(result, v);
		}
		inline Vector4f	Vector4f::FastNormalize(const Vector4f & value)
		{
			SIMD::vFloat v = LoadFloat4(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			return Vector4f(v);
		}
		inline void		Vector4f::FastNormalize(const Vector4f & value, Vector4f & result)
		{
			SIMD::vFloat v = LoadFloat4(value);
			SIMD::vFloat s = SIMD::Multiply(v, v);
			s = SIMD::AddH(s, s);
			s = SIMD::AddH(s, s);
			s = SIMD::ReciprocalSqrt(s);
			v = SIMD::Multiply(v, s);
			StoreFloat4(result, v);
		}
	#endif
		
		inline Vector4f	Vector4f::One(void)		{ return Vector4f( 1.0f,  1.0f,  1.0f,  1.0f); }
		inline Vector4f	Vector4f::UnitX(void)		{ return Vector4f( 1.0f,  0.0f,  0.0f,  0.0f); }
		inline Vector4f	Vector4f::UnitY(void)		{ return Vector4f( 0.0f,  1.0f,  0.0f,  0.0f); }
		inline Vector4f	Vector4f::UnitZ(void)		{ return Vector4f( 0.0f,  0.0f,  1.0f,  0.0f); }
		inline Vector4f	Vector4f::UnitW(void)		{ return Vector4f( 0.0f,  0.0f,  0.0f,  1.0f); }
		inline Vector4f	Vector4f::Zero(void)		{ return Vector4f( 0.0f,  0.0f,  0.0f,  0.0f); }

		inline Vector4f operator+ (const Vector4f & f1, const Vector4f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(f1);
			SIMD::vFloat v2 = LoadFloat4(f2);
			v1 = SIMD::Add(v1, v2);
			return Vector4f(v1);
		#else
			return Vector4f(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z, f1.w + f2.w);
		#endif
		}
		inline Vector4f operator- (const Vector4f & f1, const Vector4f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(f1);
			SIMD::vFloat v2 = LoadFloat4(f2);
			v1 = SIMD::Subtract(v1, v2);
			return Vector4f(v1);
		#else
			return Vector4f(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z, f1.w - f2.w);
		#endif
		}
		inline Vector4f operator* (const Vector4f & f1, const Vector4f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(f1);
			SIMD::vFloat v2 = LoadFloat4(f2);
			v1 = SIMD::Multiply(v1, v2);
			return Vector4f(v1);
		#else
			return Vector4f(f1.x * f2.x, f1.y * f2.y, f1.z * f2.z, f1.w * f2.w);
		#endif
		}
		inline Vector4f operator/ (const Vector4f & f1, const Vector4f & f2)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(f1);
			SIMD::vFloat v2 = LoadFloat4(f2);
			v1 = SIMD::Divide(v1, v2);
			return Vector4f(v1);
		#else
			return Vector4f(f1.x / f2.x, f1.y / f2.y, f1.z / f2.z, f1.w * f2.w);
		#endif
		}
		
		inline Vector4f operator* (const Vector4f & f, const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(f);
			const SIMD::vFloat v2 = SIMD::Set1(value);
			v1 = SIMD::Multiply(v1, v2);
			return Vector4f(v1);
		#else
			return Vector4f(f.x * value, f.y * value, f.z * value, f.w * value);
		#endif
		}
		inline Vector4f operator* (const float value, const Vector4f & f)
		{
			return operator*(f, value);
		}
		inline Vector4f operator/ (const Vector4f & f, const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadFloat4(f);
			const SIMD::vFloat v2 = SIMD::Set1(value);
			v1 = SIMD::Divide(v1, v2);
			return Vector4f(v1);
		#else
			return Vector4f(f.x / value, f.y / value, f.z / value, f.w / value);
		#endif
		}
		inline Vector4f operator/ (const float value, const Vector4f & f)
		{
			return operator/(f, value);
		}
	#pragma endregion
	}
}