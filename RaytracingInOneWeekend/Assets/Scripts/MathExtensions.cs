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

        // NOTE: normal is assumed to be of unit length
        public static float3 InCosineWeightedHemisphere(this Random rng, float3 normal)
        {
            // from : http://www.rorydriscoll.com/2009/01/07/better-sampling/
            // and : https://www.shadertoy.com/view/4tGcWD
            float2 uv = rng.NextFloat2();

            float radius = sqrt(uv.x);
            float theta = 2 * PI * uv.y;

            sincos(theta, out float sinTheta, out float cosTheta);
            float2 xy = radius * float2(cosTheta, sinTheta);

            float3 tangentSpaceDirection = float3(xy, sqrt(max(0, 1 - uv.x)));

            float3 up = abs(normal.y) > 0.5 ? float3(1, 0, 0) : float3(0, 1, 0);
            float3 right = normalize(cross(normal, up));
            up = cross(right, normal);

            float3x3 rotationMatrix = float3x3(right, up, normal);
            return mul(rotationMatrix, tangentSpaceDirection);
        }

        public static float3 ToFloat3(this Color c)
        {
            return float3(c.r, c.g, c.b);
        }

        public static Color ToColor(this float3 c)
        {
            return new Color(c.x, c.y, c.z);
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