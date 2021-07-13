using System.Runtime.CompilerServices;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;

namespace Util
{
    static class MathExtensions
    {
        public static float3 ToFloat3(this Color c) => float3(c.r, c.g, c.b);

        public static Color ToColor(this float3 c) => new Color(c.x, c.y, c.z);

        public static Color GetAlphaReplaced(this Color c, float alpha) => new Color(c.r, c.g, c.b, alpha);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float3 LinearToGamma(this float3 value)
        {
            value = max(value, 0);
            return max(1.055f * pow(value, 0.416666667f) - 0.055f, 0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool AlmostEquals(this float lhs, float rhs)
        {
            return abs(rhs - lhs) < 1e-6f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool2 AlmostEquals(this float2 lhs, float2 rhs)
        {
            return abs(rhs - lhs) < 1e-6f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3 AlmostEquals(this float3 lhs, float3 rhs)
        {
            return abs(rhs - lhs) < 1e-6f;
        }
    }
}