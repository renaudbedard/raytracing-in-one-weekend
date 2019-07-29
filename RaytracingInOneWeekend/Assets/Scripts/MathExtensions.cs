using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
    static class MathExtensions
    {
        public static float3 InUnitSphere(this Random rng)
        {
            // from : https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/#better-choice-of-spherical-coordinates
            float2 doubleUv = rng.NextFloat2(0, 2);
            float2 thetaPhi = float2( doubleUv.x * PI, acos(doubleUv.y - 1));
            float r = pow(rng.NextFloat(), 1 / 3.0f);
            sincos(thetaPhi, out float2 sinThetaPhi, out float2 cosThetaPhi);
            return r * float3(
                sinThetaPhi.y * cosThetaPhi.x,
                sinThetaPhi.y * sinThetaPhi.x,
                cosThetaPhi.y);
        }

        public static float2 InUnitDisk(this Random rng)
        {
            // from : https://programming.guide/random-point-within-circle.html
            float theta = rng.NextFloat(0, 2 * PI);
            float radius = sqrt(rng.NextFloat());
            sincos(theta, out float sinTheta, out float cosTheta);
            return radius * float2(cosTheta, sinTheta);
        }

        public static float3 UnitVector(this Random rng)
        {
            float z = rng.NextFloat(-1, 1);
            float a = rng.NextFloat(2 * PI);
            float r = sqrt(1.0f - z * z);
            sincos(a, out float y, out float x);
            return float3(x * r, y * r, z);
        }

        public static float3 ToFloat3(this Color c)
        {
            return float3(c.r, c.g, c.b);
        }

        public static Color ToColor(this float3 c)
        {
            return new Color(c.x, c.y, c.z);
        }

        public static uint Sum(this NativeArray<uint> values)
        {
            uint total = 0;
            foreach (uint value in values) total += value;
            return total;
        }

        public static Color GetAlphaReplaced(this Color c, float alpha)
        {
            c.a = alpha;
            return c;
        }

        public static float3 LinearToGamma(this float3 value)
        {
            value = max(value, 0);
            return max(1.055f * pow(value, 0.416666667f) - 0.055f, 0);
        }
    }
}