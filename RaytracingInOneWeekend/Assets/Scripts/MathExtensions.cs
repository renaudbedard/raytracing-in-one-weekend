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
            float u = rng.NextFloat();
            float radius = sqrt(u);
            float theta = rng.NextFloat(0, 2 * PI);
            sincos(theta, out float sinTheta, out float cosTheta);
            float3 tangentSpaceDirection = float3(radius * float2(cosTheta, sinTheta), sqrt(1 - u));

            // build an orthonormal basis from the forward (normal) vector
            // from : https://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
            float3 right, up, forward = normal;
            if(forward.z < -0.9999999f)
            {
                up = float3(0, -1, 0);
                right = float3(-1, 0, 0);
            }
            else
            {
                float a = 1 / (1 + forward.z);
                float b = -forward.x * forward.y * a;
                up = float3(1 - forward.x * forward.x * a, b, -forward.x);
                right = float3(b, 1 - forward.y * forward.y * a, -forward.y);
            }

            // transform from tangent-space to world-space
            float3x3 rotationMatrix = float3x3(right, up, forward);
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