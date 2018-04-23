namespace Ruined
{
	namespace Math
	{
	#ifdef __SIMD__
			// Loads Row0 of a Matrix into a vFloat :
			//	returns (m00, m01, m02, m03)
		inline SIMD::vFloat LoadMatrixRow0 (const Matrix & m)
		{
			return SIMD::Load(&m.m00);
		}
			// Loads Row1 of a Matrix into a vFloat :
			//	returns (m10, m11, m12, m13)
		inline SIMD::vFloat LoadMatrixRow1 (const Matrix & m)
		{
			return SIMD::Load(&m.m10);
		}
			// Loads Row2 of a Matrix into a vFloat :
			//	returns (m20, m21, m22, m23)
		inline SIMD::vFloat LoadMatrixRow2 (const Matrix & m)
		{
			return SIMD::Load(&m.m20);
		}
			// Loads Row3 of a Matrix into a vFloat :
			//	returns (m30, m31, m32, m33)
		inline SIMD::vFloat LoadMatrixRow3 (const Matrix & m)
		{
			return SIMD::Load(&m.m30);
		}

			// Stores four floating-point values into Row0 of a Matrix.
		inline void			StoreMatrixRow0(Matrix & m, SIMD::vFloat v)
		{
			SIMD::Store(&m.m00, v);
		}
			// Stores four floating-point values into Row1 of a Matrix.
		inline void			StoreMatrixRow1(Matrix & m, SIMD::vFloat v)
		{
			SIMD::Store(&m.m10, v);
		}
			// Stores four floating-point values into Row2 of a Matrix.
		inline void			StoreMatrixRow2(Matrix & m, SIMD::vFloat v)
		{
			SIMD::Store(&m.m20, v);
		}
			// Stores four floating-point values into Row3 of a Matrix.
		inline void			StoreMatrixRow3(Matrix & m, SIMD::vFloat v)
		{
			SIMD::Store(&m.m30, v);
		}
	#endif

		inline Matrix::Matrix (void)
		{
			//m00 = m11 = m22 = m33 = 1.0f;
			//m01 = m02 = m03 =
			//m10 = m12 = m23 =
			//m20 = m21 = m23 =
			//m30 = m31 = m32 = 0.0f;
		}
		inline Matrix::Matrix (	float const m00, float const m01, float const m02, float const m03,
								float const m10, float const m11, float const m12, float const m13,
								float const m20, float const m21, float const m22, float const m23,
								float const m30, float const m31, float const m32, float const m33)
				:	m00(m00), m01(m01), m02(m02), m03(m03),
					m10(m10), m11(m11), m12(m12), m13(m13),
					m20(m20), m21(m21), m22(m22), m23(m23),
					m30(m30), m31(m31), m32(m32), m33(m33) {}
		inline Matrix::Matrix (const float * const pArray)
		{
			m00 = pArray[0];
			m01 = pArray[1];
			m02 = pArray[2];
			m03 = pArray[3];

			m10 = pArray[4];
			m11 = pArray[5];
			m12 = pArray[6];
			m13 = pArray[7];

			m20 = pArray[8];
			m21 = pArray[9];
			m22 = pArray[10];
			m23 = pArray[11];

			m30 = pArray[12];
			m31 = pArray[13];
			m32 = pArray[14];
			m33 = pArray[15];
		}
			
		inline bool		Matrix::operator== (const Matrix & other) const
		{
			return	other.m00 == m00 && other.m01 == m01 && other.m02 == m02 && other.m03 == m03 &&
					other.m10 == m10 && other.m11 == m11 && other.m12 == m12 && other.m13 == m13 &&
					other.m20 == m20 && other.m21 == m21 && other.m22 == m22 && other.m23 == m23 &&
					other.m30 == m30 && other.m31 == m31 && other.m32 == m32 && other.m33 == m33;
		}
		inline bool		Matrix::operator!= (const Matrix & other) const
		{
			return !(*this == other);
		}

		inline Matrix &	Matrix::operator= (const Matrix & other)
		{
			m00 = other.m00;
			m01 = other.m01;
			m02 = other.m02;
			m03 = other.m03;

			m10 = other.m10;
			m11 = other.m11;
			m12 = other.m12;
			m13 = other.m13;

			m20 = other.m20;
			m21 = other.m21;
			m22 = other.m22;
			m23 = other.m23;

			m30 = other.m30;
			m31 = other.m31;
			m32 = other.m32;
			m33 = other.m33;

			return *this;
		}

		inline Matrix	Matrix::operator+ () const { return *this; }
		inline Matrix	Matrix::operator- () const
		{
			return Matrix(	-m00, -m01, -m02, -m03,
							-m10, -m11, -m12, -m13,
							-m20, -m21, -m22, -m23,
							-m30, -m31, -m32, -m33);
		}

		inline Matrix &	Matrix::operator+= (const Matrix & other)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadMatrixRow0(*this);
			SIMD::vFloat v2 = LoadMatrixRow0(other);
			SIMD::vFloat vR = SIMD::Add(v1, v2);
			StoreMatrixRow0(*this, vR);

			v1 = LoadMatrixRow1(*this);
			v2 = LoadMatrixRow1(other);
			vR = SIMD::Add(v1, v2);
			StoreMatrixRow1(*this, vR);

			v1 = LoadMatrixRow2(*this);
			v2 = LoadMatrixRow2(other);
			vR = SIMD::Add(v1, v2);
			StoreMatrixRow2(*this, vR);

			v1 = LoadMatrixRow3(*this);
			v2 = LoadMatrixRow3(other);
			vR = SIMD::Add(v1, v2);
			StoreMatrixRow3(*this, vR);
		#else
			m00 += other.m00;
			m01 += other.m01;
			m02 += other.m02;
			m03 += other.m03;

			m10 += other.m10;
			m11 += other.m11;
			m12 += other.m12;
			m13 += other.m13;

			m20 += other.m20;
			m21 += other.m21;
			m22 += other.m22;
			m23 += other.m23;

			m30 += other.m30;
			m31 += other.m31;
			m32 += other.m32;
			m33 += other.m33;
		#endif
			return *this;
		}
		inline Matrix &	Matrix::operator-= (const Matrix & other)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadMatrixRow0(*this);
			SIMD::vFloat v2 = LoadMatrixRow0(other);
			SIMD::vFloat vR = SIMD::Subtract(v1, v2);
			StoreMatrixRow0(*this, vR);

			v1 = LoadMatrixRow1(*this);
			v2 = LoadMatrixRow1(other);
			vR = SIMD::Subtract(v1, v2);
			StoreMatrixRow1(*this, vR);

			v1 = LoadMatrixRow2(*this);
			v2 = LoadMatrixRow2(other);
			vR = SIMD::Subtract(v1, v2);
			StoreMatrixRow2(*this, vR);

			v1 = LoadMatrixRow3(*this);
			v2 = LoadMatrixRow3(other);
			vR = SIMD::Subtract(v1, v2);
			StoreMatrixRow3(*this, vR);
		#else
			m00 -= other.m00;
			m01 -= other.m01;
			m02 -= other.m02;
			m03 -= other.m03;

			m10 -= other.m10;
			m11 -= other.m11;
			m12 -= other.m12;
			m13 -= other.m13;

			m20 -= other.m20;
			m21 -= other.m21;
			m22 -= other.m22;
			m23 -= other.m23;

			m30 -= other.m30;
			m31 -= other.m31;
			m32 -= other.m32;
			m33 -= other.m33;
		#endif
			return *this;
		}
		inline Matrix &	Matrix::operator*= (const Matrix & other)
		{
		#ifdef __SIMD__
			const SIMD::vFloat v0 = LoadMatrixRow0(other);
			const SIMD::vFloat v1 = LoadMatrixRow1(other);
			const SIMD::vFloat v2 = LoadMatrixRow2(other);
			const SIMD::vFloat v3 = LoadMatrixRow3(other);

			SIMD::vFloat t1 = SIMD::Set1(m00);
			SIMD::vFloat t2 = SIMD::Multiply(v0, t1);
			t1 = SIMD::Set1(m01);
			t2 = SIMD::Add(SIMD::Multiply(v1, t1), t2);
			t1 = SIMD::Set1(m02);
			t2 = SIMD::Add(SIMD::Multiply(v2, t1), t2);
			t1 = SIMD::Set1(m03);
			t2 = SIMD::Add(SIMD::Multiply(v3, t1), t2);
			StoreMatrixRow0(*this, t2);

			t1 = SIMD::Set1(m10);
			t2 = SIMD::Multiply(v0, t1);
			t1 = SIMD::Set1(m11);
			t2 = SIMD::Add(SIMD::Multiply(v1, t1), t2);
			t1 = SIMD::Set1(m12);
			t2 = SIMD::Add(SIMD::Multiply(v2, t1), t2);
			t1 = SIMD::Set1(m13);
			t2 = SIMD::Add(SIMD::Multiply(v3, t1), t2);
			StoreMatrixRow1(*this, t2);

			t1 = SIMD::Set1(m20);
			t2 = SIMD::Multiply(v0, t1);
			t1 = SIMD::Set1(m21);
			t2 = SIMD::Add(SIMD::Multiply(v1, t1), t2);
			t1 = SIMD::Set1(m22);
			t2 = SIMD::Add(SIMD::Multiply(v2, t1), t2);
			t1 = SIMD::Set1(m23);
			t2 = SIMD::Add(SIMD::Multiply(v3, t1), t2);
			StoreMatrixRow2(*this, t2);

			t1 = SIMD::Set1(m30);
			t2 = SIMD::Multiply(v0, t1);
			t1 = SIMD::Set1(m31);
			t2 = SIMD::Add(SIMD::Multiply(v1, t1), t2);
			t1 = SIMD::Set1(m32);
			t2 = SIMD::Add(SIMD::Multiply(v2, t1), t2);
			t1 = SIMD::Set1(m33);
			t2 = SIMD::Add(SIMD::Multiply(v3, t1), t2);
			StoreMatrixRow3(*this, t2);
		#else
			*this = Matrix
			(m00 * other.m00	+	m01 * other.m10	+	m02 * other.m20	+	m03 * other.m30,
			m00 * other.m01	+	m01 * other.m11	+	m02 * other.m21	+	m03 * other.m31,
			m00 * other.m02	+	m01 * other.m12	+	m02 * other.m22	+	m03 * other.m32,
			m00 * other.m03	+	m01 * other.m13	+	m02 * other.m23	+	m03 * other.m33,

			m10 * other.m00	+	m11 * other.m10	+	m12 * other.m20	+	m13 * other.m30,
			m10 * other.m01	+	m11 * other.m11	+	m12 * other.m21	+	m13 * other.m31,
			m10 * other.m02	+	m11 * other.m12	+	m12 * other.m22	+	m13 * other.m32,
			m10 * other.m03	+	m11 * other.m13	+	m12 * other.m23	+	m13 * other.m33,

			m20 * other.m00	+	m21 * other.m10	+	m22 * other.m20	+	m23 * other.m30,
			m20 * other.m01	+	m21 * other.m11	+	m22 * other.m21	+	m23 * other.m31,
			m20 * other.m02	+	m21 * other.m12	+	m22 * other.m22	+	m23 * other.m32,
			m20 * other.m03	+	m21 * other.m13	+	m22 * other.m23	+	m23 * other.m33,
			
			m30 * other.m00	+	m31 * other.m10	+	m32 * other.m20	+	m33 * other.m30,
			m30 * other.m01	+	m31 * other.m11	+	m32 * other.m21	+	m33 * other.m31,
			m30 * other.m02	+	m31 * other.m12	+	m32 * other.m22	+	m33 * other.m32,
			m30 * other.m03	+	m31 * other.m13	+	m32 * other.m23	+	m33 * other.m33);
		#endif
			return *this;
		}
		
		inline Matrix &	Matrix::operator*= (const float value)
		{
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadMatrixRow0(*this);
			const SIMD::vFloat vS = SIMD::Set1(value);
			SIMD::vFloat vR = SIMD::Multiply(v1, vS);
			StoreMatrixRow0(*this, vR);

			v1 = LoadMatrixRow1(*this);
			vR = SIMD::Multiply(v1, vS);
			StoreMatrixRow1(*this, vR);

			v1 = LoadMatrixRow2(*this);
			vR = SIMD::Multiply(v1, vS);
			StoreMatrixRow2(*this, vR);

			v1 = LoadMatrixRow3(*this);
			vR = SIMD::Multiply(v1, vS);
			StoreMatrixRow3(*this, vR);
		#else
			m00 *= value;
			m01 *= value;
			m02 *= value;
			m03 *= value;

			m10 *= value;
			m11 *= value;
			m12 *= value;
			m13 *= value;

			m20 *= value;
			m21 *= value;
			m22 *= value;
			m23 *= value;

			m30 *= value;
			m31 *= value;
			m32 *= value;
			m33 *= value;
		#endif
			return *this;
		}
		inline Matrix &	Matrix::operator/= (const float value)
		{
			assert(value != 0.0f);
		#ifdef __SIMD__
			SIMD::vFloat v1 = LoadMatrixRow0(*this);
			const SIMD::vFloat vS = SIMD::Set1(value);
			SIMD::vFloat vR = SIMD::Divide(v1, vS);
			StoreMatrixRow0(*this, vR);

			v1 = LoadMatrixRow1(*this);
			vR = SIMD::Divide(v1, vS);
			StoreMatrixRow1(*this, vR);

			v1 = LoadMatrixRow2(*this);
			vR = SIMD::Divide(v1, vS);
			StoreMatrixRow2(*this, vR);

			v1 = LoadMatrixRow3(*this);
			vR = SIMD::Divide(v1, vS);
			StoreMatrixRow3(*this, vR);
		#else
			m00 /= value;
			m01 /= value;
			m02 /= value;
			m03 /= value;

			m10 /= value;
			m11 /= value;
			m12 /= value;
			m13 /= value;

			m20 /= value;
			m21 /= value;
			m22 /= value;
			m23 /= value;

			m30 /= value;
			m31 /= value;
			m32 /= value;
			m33 /= value;
		#endif
			return *this;
		}

		inline Matrix	Matrix::CreateLookAt(Vector3f const cameraPosition, Vector3f const cameraTarget, Vector3f const cameraUp)
		{
			Matrix result;
			CreateLookAt(cameraPosition, cameraTarget, cameraUp, result);
			return result;
		}
		inline void		Matrix::CreateLookAt(Vector3f const cameraPosition, Vector3f const cameraTarget, Vector3f const cameraUp, Matrix & result)
		{
			Vector3f zAxis = Vector3f::Normalize(cameraTarget - cameraPosition);
			Vector3f xAxis = Vector3f::Normalize(Vector3f::Cross(cameraUp, zAxis));
			Vector3f yAxis = Vector3f::Cross(zAxis, xAxis);

			result.m00 = xAxis.x;
			result.m01 = yAxis.x;
			result.m02 = zAxis.x;
			result.m03 = 0.0f;

			result.m10 = xAxis.y;
			result.m11 = yAxis.y;
			result.m12 = zAxis.y;
			result.m13 = 0.0f;

			result.m20 = xAxis.z;
			result.m21 = yAxis.z;
			result.m22 = zAxis.z;
			result.m23 = 0.0f;

			result.m30 = -(Vector3f::Dot(xAxis, cameraPosition));
			result.m31 = -(Vector3f::Dot(yAxis, cameraPosition));
			result.m32 = -(Vector3f::Dot(zAxis, cameraPosition));
			result.m33 = 1.0f;
		}

		inline Matrix	Matrix::CreateOrthographic(float const width, float const height, float const nearZ, float const farZ)
		{
			Matrix result;
			CreateOrthographic(width, height, nearZ, farZ, result);
			return result;
		}
		inline void		Matrix::CreateOrthographic(float const width, float const height, float const nearZ, float const farZ, Matrix & result)
		{
			float rangeZ	= farZ / (nearZ - farZ);

			result.m00 = 2.0f / width;
			result.m01 = 0.0f;
			result.m02 = 0.0f;
			result.m03 = 0.0f;

			result.m10 = 0.0f;
			result.m11 = 2.0f / height;
			result.m12 = 0.0f;
			result.m13 = 0.0f;

			result.m20 = 0.0f;
			result.m21 = 0.0f;
			result.m22 = rangeZ;
			result.m23 = 0.0f;

			result.m30 = 0.0f;
			result.m31 = 0.0f;
			result.m32 = rangeZ * nearZ;
			result.m33 = 0.0f;
		}

		inline Matrix	Matrix::CreatePerspective(float const width, const float height, float const nearZ, float const farZ)
		{
			Matrix result;
			CreatePerspective(width, height, nearZ, farZ, result);
			return result;
		}
		inline void		Matrix::CreatePerspective(float const width, const float height, float const nearZ, float const farZ, Matrix & result)
		{
			float twoNear	= nearZ + nearZ;
			float rangeZ	= farZ / (nearZ - farZ);

			result.m00 = twoNear / width;
			result.m01 = 0.0f; 
			result.m02 = 0.0f; 
			result.m03 = 0.0f;

			result.m10 = 0.0f;
			result.m11 = twoNear / height;
			result.m12 = 0.0f; 
			result.m13 = 0.0f;

			result.m20 = 0.0f; 
			result.m21 = 0.0f;
			result.m22 = rangeZ;
			result.m23 = -1.0f;
			
			result.m30 = 0.0f;
			result.m31 = 0.0f;
			result.m32 = rangeZ * nearZ;
			result.m33 = 0.0f;
		}

		inline Matrix	Matrix::CreatePerspectiveFov(float const fovAngle, float const aspectRatio, float const nearZ, float const farZ)
		{
			Matrix result;
			CreatePerspectiveFov(fovAngle, aspectRatio, nearZ, farZ, result);
			return result;
		}
		inline void		Matrix::CreatePerspectiveFov(float const fovAngle, float const aspectRatio, float const nearZ, float const farZ, Matrix & result)
		{
			float CosFov	= cosf(fovAngle * 0.5f);
			float SinFov	= sinf(fovAngle * 0.5f);

			float Height	= CosFov / SinFov;
			float Width		= Height / aspectRatio;
			float rangeZ	= farZ / (farZ - nearZ);

			result.m00 = Width;
			result.m01 = 0.0f;
			result.m02 = 0.0f;
			result.m03 = 0.0f;

			result.m10 = 0.0f;
			result.m11 = Height;
			result.m12 = 0.0f;
			result.m13 = 0.0f;

			result.m20 = 0.0f;
			result.m21 = 0.0f;
			result.m22 = rangeZ;
			result.m23 = 1.0f;

			result.m30 = 0.0f;
			result.m31 = 0.0f;
			result.m32 = -(rangeZ * nearZ);
			result.m33 = 0.0f;
		}	

		inline Matrix	Matrix::CreateRotationX(float radians)
		{
			Matrix result;
			CreateRotationX(radians, result);
			return result;
		}
		inline Matrix	Matrix::CreateRotationY(float radians)
		{
			Matrix result;
			CreateRotationY(radians, result);
			return result;
		}
		inline Matrix	Matrix::CreateRotationZ(float radians)
		{
			Matrix result;
			CreateRotationZ(radians, result);
			return result;
		}
		inline void		Matrix::CreateRotationX(float const radians, Matrix & result)
		{
			float CosTheta = cos(radians);
			float SinTheta = sin(radians);

			result.m00 = 1.0f;
			result.m01 = 0.0f;
			result.m02 = 0.0f;
			result.m03 = 0.0f;

			result.m10 = 0.0f;
			result.m11 = CosTheta;
			result.m12 = -SinTheta;
			result.m13 = 0.0f;

			result.m20 = 0.0f;
			result.m21 = SinTheta;
			result.m22 = CosTheta;
			result.m23 = 0.0f;

			result.m30 = 0.0f;
			result.m31 = 0.0f;
			result.m32 = 0.0f;
			result.m33 = 1.0f;
		}
		inline void		Matrix::CreateRotationY(float const radians, Matrix & result)
		{
			float CosTheta = cos(radians);
			float SinTheta = sin(radians);

			result.m00 = CosTheta;
			result.m01 = 0.0f;
			result.m02 = SinTheta;
			result.m03 = 0.0f;

			result.m10 = 0.0f;
			result.m11 = 1.0f;
			result.m12 = 0.0f;
			result.m13 = 0.0f;

			result.m20 = -SinTheta;
			result.m21 = 0.0f;
			result.m22 = CosTheta;
			result.m23 = 0.0f;

			result.m30 = 0.0f;
			result.m31 = 0.0f;
			result.m32 = 0.0f;
			result.m33 = 1.0f;
		}
		inline void		Matrix::CreateRotationZ(float const radians, Matrix & result)
		{
			float CosTheta = cos(radians);
			float SinTheta = sin(radians);

			result.m00 = CosTheta;
			result.m01 = -SinTheta;
			result.m02 = 0.0f;
			result.m03 = 0.0f;

			result.m10 = SinTheta;
			result.m11 = CosTheta;
			result.m12 = 0.0f;
			result.m13 = 0.0f;

			result.m20 = 0.0f;
			result.m21 = 0.0f;
			result.m22 = 1.0f;
			result.m23 = 0.0f;

			result.m30 = 0.0f;
			result.m31 = 0.0f;
			result.m32 = 0.0f;
			result.m33 = 1.0f;
		}

		inline Matrix	Matrix::CreateScale(float const scale)
		{
			Matrix result;
			CreateScale(scale, result);
			return result;
		}
		inline Matrix	Matrix::CreateScale(float3 const scales)
		{
			Matrix result;
			CreateScale(scales, result);
			return result;
		}
		inline Matrix	Matrix::CreateScale(float const xScale, float const yScale, float const zScale)
		{
			Matrix result;
			CreateScale(xScale, yScale, zScale, result);
			return result;
		}
		inline void		Matrix::CreateScale(float const scale, Matrix & result)
		{
			result = Matrix(
				scale,	0.0f,	0.0f,	0.0f,
				0.0f,	scale,	0.0f,	0.0f,
				0.0f,	0.0f,	scale,	0.0f,
				0.0f,	0.0f,	0.0f,	1.0f);
		}
		inline void		Matrix::CreateScale(float3 const & scales, Matrix & result)
		{
			result = Matrix(
				scales.x,	0.0f,		0.0f,		0.0f,
				0.0f,		scales.y,	0.0f,		0.0f,
				0.0f,		0.0f,		scales.z,	0.0f,
				0.0f,		0.0f,		0.0f,		1.0f);
		}
		inline void		Matrix::CreateScale(float const xScale, float const yScale, float const zScale, Matrix & result)
		{
			result = Matrix(
				xScale,	0.0f,	0.0f,	0.0f,
				0.0f,	yScale,	0.0f,	0.0f,
				0.0f,	0.0f,	zScale,	0.0f,
				0.0f,	0.0f,	0.0f,	1.0f);
		}

		inline Matrix	Matrix::CreateTranslation(float3 const position)
		{
			Matrix result;
			CreateTranslation(position, result);
			return result;
		}
		inline Matrix	Matrix::CreateTranslation(float const xPosition, float const yPosition, float const zPosition)
		{
			Matrix result;
			CreateTranslation(xPosition, yPosition, zPosition, result);
			return result;
		}
		inline void		Matrix::CreateTranslation(float3 const & position, Matrix & result)
		{
			result = Matrix(
				1.0f,	0.0f,	0.0f,	position.x,
				0.0f,	1.0f,	0.0f,	position.y,
				0.0f,	0.0f,	1.0f,	position.z,
				0.0f,	0.0f,	0.0f,	1.0f);
		}
		inline void		Matrix::CreateTranslation(float const xPosition, float const yPosition, float const zPosition, Matrix & result)
		{
			result = Matrix(
				1.0f,	0.0f,	0.0f,	xPosition,
				0.0f,	1.0f,	0.0f,	yPosition,
				0.0f,	0.0f,	1.0f,	zPosition,
				0.0f,	0.0f,	0.0f,	1.0f);
		}

		inline Matrix	Matrix::Identity()
		{
			return Matrix(	1.0f, 0.0f, 0.0f, 0.0f,
							0.0f, 1.0f, 0.0f, 0.0f,
							0.0f, 0.0f, 1.0f, 0.0f,
							0.0f, 0.0f, 0.0f, 1.0f);
		}

		inline Matrix	Matrix::Transpose(Matrix & matrix)
		{
			Matrix result;
			Transpose(matrix, result);
			return result;
		}
		inline void	Matrix::Transpose(Matrix const & matrix, Matrix & result)
		{
			result.m00 = matrix.m00;
			result.m01 = matrix.m10;
			result.m02 = matrix.m20;
			result.m03 = matrix.m30;

			result.m10 = matrix.m01;
			result.m11 = matrix.m11;
			result.m12 = matrix.m21;
			result.m13 = matrix.m31;

			result.m20 = matrix.m02;
			result.m21 = matrix.m12;
			result.m22 = matrix.m22;
			result.m23 = matrix.m32;

			result.m30 = matrix.m03;
			result.m31 = matrix.m13;
			result.m32 = matrix.m23;
			result.m33 = matrix.m33;
		}

		inline Matrix operator+ (const Matrix & a, const Matrix & b)
		{
		#ifdef __SIMD__
			Matrix mR;
			SIMD::vFloat v1 = LoadMatrixRow0(a);
			SIMD::vFloat v2 = LoadMatrixRow0(b);
			SIMD::vFloat vR = SIMD::Add(v1, v2);
			StoreMatrixRow0(mR, vR);

			v1 = LoadMatrixRow1(a);
			v2 = LoadMatrixRow1(b);
			vR = SIMD::Add(v1, v2);
			StoreMatrixRow1(mR, vR);

			v1 = LoadMatrixRow2(a);
			v2 = LoadMatrixRow2(b);
			vR = SIMD::Add(v1, v2);
			StoreMatrixRow2(mR, vR);

			v1 = LoadMatrixRow3(a);
			v2 = LoadMatrixRow3(b);
			vR = SIMD::Add(v1, v2);
			StoreMatrixRow3(mR, vR);

			return mR;
		#else
			return Matrix(	a.m00 + b.m00, a.m01 + b.m01, a.m02 + b.m02, a.m03 + b.m03,
							a.m10 + b.m10, a.m11 + b.m11, a.m12 + b.m12, a.m13 + b.m23,
							a.m20 + b.m20, a.m21 + b.m21, a.m22 + b.m22, a.m23 + b.m23,
							a.m30 + b.m30, a.m31 + b.m31, a.m32 + b.m32, a.m33 + b.m33);
		#endif
		}
		inline Matrix operator- (const Matrix & a, const Matrix & b)
		{
		#ifdef __SIMD__
			Matrix mR;
			SIMD::vFloat v1 = LoadMatrixRow0(a);
			SIMD::vFloat v2 = LoadMatrixRow0(b);
			SIMD::vFloat vR = SIMD::Subtract(v1, v2);
			StoreMatrixRow0(mR, vR);

			v1 = LoadMatrixRow1(a);
			v2 = LoadMatrixRow1(b);
			vR = SIMD::Subtract(v1, v2);
			StoreMatrixRow1(mR, vR);

			v1 = LoadMatrixRow2(a);
			v2 = LoadMatrixRow2(b);
			vR = SIMD::Subtract(v1, v2);
			StoreMatrixRow2(mR, vR);

			v1 = LoadMatrixRow3(a);
			v2 = LoadMatrixRow3(b);
			vR = SIMD::Subtract(v1, v2);
			StoreMatrixRow3(mR, vR);

			return mR;
		#else
			return Matrix(	a.m00 - b.m00, a.m01 - b.m01, a.m02 - b.m02, a.m03 - b.m03,
							a.m10 - b.m10, a.m11 - b.m11, a.m12 - b.m12, a.m13 - b.m23,
							a.m20 - b.m20, a.m21 - b.m21, a.m22 - b.m22, a.m23 - b.m23,
							a.m30 - b.m30, a.m31 - b.m31, a.m32 - b.m32, a.m33 - b.m33);
		#endif
		}

		inline Matrix operator* (const Matrix & a, const Matrix & b)
		{
		#ifdef __SIMD__
			Matrix mR;

			const SIMD::vFloat v0 = LoadMatrixRow0(b);
			const SIMD::vFloat v1 = LoadMatrixRow1(b);
			const SIMD::vFloat v2 = LoadMatrixRow2(b);
			const SIMD::vFloat v3 = LoadMatrixRow3(b);

			SIMD::vFloat t1 = SIMD::Set1(a.m00);
			SIMD::vFloat t2 = SIMD::Multiply(v0, t1);
			t1 = SIMD::Set1(a.m01);
			t2 = SIMD::Add(SIMD::Multiply(v1, t1), t2);
			t1 = SIMD::Set1(a.m02);
			t2 = SIMD::Add(SIMD::Multiply(v2, t1), t2);
			t1 = SIMD::Set1(a.m03);
			t2 = SIMD::Add(SIMD::Multiply(v3, t1), t2);
			StoreMatrixRow0(mR, t2);

			t1 = SIMD::Set1(a.m10);
			t2 = SIMD::Multiply(v0, t1);
			t1 = SIMD::Set1(a.m11);
			t2 = SIMD::Add(SIMD::Multiply(v1, t1), t2);
			t1 = SIMD::Set1(a.m12);
			t2 = SIMD::Add(SIMD::Multiply(v2, t1), t2);
			t1 = SIMD::Set1(a.m13);
			t2 = SIMD::Add(SIMD::Multiply(v3, t1), t2);
			StoreMatrixRow1(mR, t2);

			t1 = SIMD::Set1(a.m20);
			t2 = SIMD::Multiply(v0, t1);
			t1 = SIMD::Set1(a.m21);
			t2 = SIMD::Add(SIMD::Multiply(v1, t1), t2);
			t1 = SIMD::Set1(a.m22);
			t2 = SIMD::Add(SIMD::Multiply(v2, t1), t2);
			t1 = SIMD::Set1(a.m23);
			t2 = SIMD::Add(SIMD::Multiply(v3, t1), t2);
			StoreMatrixRow2(mR, t2);

			t1 = SIMD::Set1(a.m30);
			t2 = SIMD::Multiply(v0, t1);
			t1 = SIMD::Set1(a.m31);
			t2 = SIMD::Add(SIMD::Multiply(v1, t1), t2);
			t1 = SIMD::Set1(a.m32);
			t2 = SIMD::Add(SIMD::Multiply(v2, t1), t2);
			t1 = SIMD::Set1(a.m33);
			t2 = SIMD::Add(SIMD::Multiply(v3, t1), t2);
			StoreMatrixRow3(mR, t2);

			return mR;
		#else
			return Matrix
			(a.m00 * b.m00	+	a.m01 * b.m10	+	a.m02 * b.m20	+	a.m03 * b.m30,
			a.m00 * b.m01	+	a.m01 * b.m11	+	a.m02 * b.m21	+	a.m03 * b.m31,
			a.m00 * b.m02	+	a.m01 * b.m12	+	a.m02 * b.m22	+	a.m03 * b.m32,
			a.m00 * b.m03	+	a.m01 * b.m13	+	a.m02 * b.m23	+	a.m03 * b.m33,

			a.m10 * b.m00	+	a.m11 * b.m10	+	a.m12 * b.m20	+	a.m13 * b.m30,
			a.m10 * b.m01	+	a.m11 * b.m11	+	a.m12 * b.m21	+	a.m13 * b.m31,
			a.m10 * b.m02	+	a.m11 * b.m12	+	a.m12 * b.m22	+	a.m13 * b.m32,
			a.m10 * b.m03	+	a.m11 * b.m13	+	a.m12 * b.m23	+	a.m13 * b.m33,

			a.m20 * b.m00	+	a.m21 * b.m10	+	a.m22 * b.m20	+	a.m23 * b.m30,
			a.m20 * b.m01	+	a.m21 * b.m11	+	a.m22 * b.m21	+	a.m23 * b.m31,
			a.m20 * b.m02	+	a.m21 * b.m12	+	a.m22 * b.m22	+	a.m23 * b.m32,
			a.m20 * b.m03	+	a.m21 * b.m13	+	a.m22 * b.m23	+	a.m23 * b.m33,
			
			a.m30 * b.m00	+	a.m31 * b.m10	+	a.m32 * b.m20	+	a.m33 * b.m30,
			a.m30 * b.m01	+	a.m31 * b.m11	+	a.m32 * b.m21	+	a.m33 * b.m31,
			a.m30 * b.m02	+	a.m31 * b.m12	+	a.m32 * b.m22	+	a.m33 * b.m32,
			a.m30 * b.m03	+	a.m31 * b.m13	+	a.m32 * b.m23	+	a.m33 * b.m33);
		#endif
		}
		
		inline Matrix operator* (const Matrix & m, const float value)
		{
		#ifdef __SIMD__
			Matrix mR;
			SIMD::vFloat v1 = LoadMatrixRow0(m);
			const SIMD::vFloat vS = SIMD::Set1(value);
			SIMD::vFloat vR = SIMD::Multiply(v1, vS);
			StoreMatrixRow0(mR, vR);

			v1 = LoadMatrixRow1(m);
			vR = SIMD::Multiply(v1, vS);
			StoreMatrixRow1(mR, vR);

			v1 = LoadMatrixRow2(m);
			vR = SIMD::Multiply(v1, vS);
			StoreMatrixRow2(mR, vR);

			v1 = LoadMatrixRow3(m);
			vR = SIMD::Multiply(v1, vS);
			StoreMatrixRow3(mR, vR);
			return mR;
		#else
			return Matrix(	m.m00 * value, m.m01 * value, m.m02 * value, m.m03 * value,
							m.m10 * value, m.m11 * value, m.m12 * value, m.m13 * value,
							m.m20 * value, m.m21 * value, m.m22 * value, m.m23 * value,
							m.m30 * value, m.m31 * value, m.m32 * value, m.m33 * value);
		#endif
		}
		inline Matrix operator* (const float value, const Matrix & m)
		{
			return operator*(m, value);
		}
		inline Matrix operator/ (const Matrix & m, const float value)
		{
		#ifdef __SIMD__
			Matrix mR;
			SIMD::vFloat v1 = LoadMatrixRow0(m);
			const SIMD::vFloat vS = SIMD::Set1(value);
			SIMD::vFloat vR = SIMD::Divide(v1, vS);
			StoreMatrixRow0(mR, vR);

			v1 = LoadMatrixRow1(m);
			vR = SIMD::Divide(v1, vS);
			StoreMatrixRow1(mR, vR);

			v1 = LoadMatrixRow2(m);
			vR = SIMD::Divide(v1, vS);
			StoreMatrixRow2(mR, vR);

			v1 = LoadMatrixRow3(m);
			vR = SIMD::Divide(v1, vS);
			StoreMatrixRow3(mR, vR);

			return mR;
		#else
			return Matrix(	m.m00 / value, m.m01 / value, m.m02 / value, m.m03 / value,
							m.m10 / value, m.m11 / value, m.m12 / value, m.m13 / value,
							m.m20 / value, m.m21 / value, m.m22 / value, m.m23 / value,
							m.m30 / value, m.m31 / value, m.m32 / value, m.m33 / value);
		#endif
		}
	}
}