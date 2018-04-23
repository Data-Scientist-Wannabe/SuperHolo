#pragma once

#include <stdint.h>

namespace Ruined
{
        namespace Math
        {
                template < typename T > struct type2
                {
                        union
                        {
                                T   data[2];

                                struct
                                {
                                        T x;
                                        T y;
                                };
                        };

                        bool operator==(const type2 & other) const
                        {	
                                return (other.x == x && other.y == y);
                        }
                        bool operator!=(const type2 & other) const
                        {
                                return !(*this == other);
                        }
                };

                template < typename T > struct type3
                {
                        union
                        {
                                T   data[3];

                                type2<T>  xy;
                                struct
                                {
                                        T   x;
                                        union 
                                        {
                                                type2<T>  yz;
                                                struct
                                                {
                                                        T   y;
                                                        T   z;
                                                };
                                        };
                                };
                        };

                        bool operator==(const type3 & other) const
                        {	
                                return (other.x == x && other.y == y && other.z == z);
                        }
                        bool operator!=(const type3 & other) const
                        {
                                return !(*this == other);
                        }
                };

                template < typename T > struct type4
                {
                        union
                        {
                                T   data[4];

                                type2<T>  xy;
                                type3<T>  xyz;
                                struct
                                {
                                        T   x;
                                        union 
                                        {
                                                type2<T>  yz;
                                                type3<T>  yzw;
                                                struct
                                                {
                                                        T   y;
                                                        union
                                                        {
                                                                type2<T>  zw;
                                                                struct
                                                                {
                                                                        T   z;
                                                                        T   w;
                                                                };
                                                        };
                                                };
                                        };
                                };
                        };

                        bool operator==(const type4 & other) const
                        {
                                return (other.x == x && other.y == y && other.z == z && other.w == w);
                        }
                        bool operator!=(const type4 & other) const
                        {
                                return !(*this == other);
                        }
                };

                template < typename T > type2<T> Type2 ( T x, T y)
                {
                        return { x, y };
                }

                template < typename T > type3<T> Type3 ( T x, T y, T z)
                {
                        return { x, y, z };
                }

                template < typename T > type4<T> Type4 ( T x, T y, T z, T w)
                {
                        return { x, y, z, w };
                }

                typedef type2<float> float2;
                typedef type3<float> float3;
                typedef type4<float> float4;

                typedef type2<double> double2;
                typedef type3<double> double3;
                typedef type4<double> double4;

                typedef type2<int8_t> char2;
                typedef type3<int8_t> char3;
                typedef type4<int8_t> char4;

                typedef type2<uint8_t> uchar2;
                typedef type3<uint8_t> uchar3;
                typedef type4<uint8_t> uchar4;

                typedef type2<int16_t> short2;
                typedef type3<int16_t> short3;
                typedef type4<int16_t> short4;

                typedef type2<uint16_t> ushort2;
                typedef type3<uint16_t> ushort3;
                typedef type4<uint16_t> ushort4;
        }
}