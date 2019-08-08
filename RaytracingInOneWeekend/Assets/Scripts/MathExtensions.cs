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
            // uniform sampling of a cosine-weighted hemisphere
            // from : https://cg.informatik.uni-freiburg.de/course_notes/graphics2_08_renderingEquation.pdf (inversion method, page 47)
            // same algorithm used here : http://www.rorydriscoll.com/2009/01/07/better-sampling/
            float u = rng.NextFloat();
            float radius = sqrt(u);
            float theta = rng.NextFloat(0, 2 * PI);
            sincos(theta, out float sinTheta, out float cosTheta);
            float3 tangentSpaceDirection = float3(radius * float2(cosTheta, sinTheta), sqrt(1 - u));

            // build an orthonormal basis from the forward (normal) vector
            // from listing 3 in : https://graphics.pixar.com/library/OrthonormalB/paper.pdf
            float s = normal.z >= 0.0f ? 1.0f : -1.0f;
            float a = -1 / (s + normal.z);
            float b = normal.x * normal.y * a;
            float3 tangent = float3(1 + s * normal.x * normal.x * a, s * b, -s * normal.x);
            float3 bitangent = float3(b, s + normal.y * normal.y * a, -normal.y);

            // transform from tangent-space to world-space
            float3x3 rotationMatrix = float3x3(bitangent, tangent, normal);
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