using System.Runtime.CompilerServices;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
    static class MathExtensions
    {
        public static float2 InUnitDisk(this ref Random rng)
        {
            // from : https://programming.guide/random-point-within-circle.html
            float theta = rng.NextFloat(0, 2 * PI);
            float radius = sqrt(rng.NextFloat());
            sincos(theta, out float sinTheta, out float cosTheta);
            return radius * float2(cosTheta, sinTheta);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void GetOrthonormalBasis(float3 normal, out float3 tangent, out float3 bitangent)
        {
            // Corrected Frisvad method
            // from listing 3 in : https://graphics.pixar.com/library/OrthonormalB/paper.pdf
            float s = normal.z >= 0 ? 1.0f : -1.0f;
            float a = -1 / (s + normal.z);
            float b = normal.x * normal.y * a;
            tangent = float3(1 + s * normal.x * normal.x * a, s * b, -s * normal.x);
            bitangent = float3(b, s + normal.y * normal.y * a, -normal.y);
        }

        public static float3 OnUniformHemisphere(this ref Random rng, float3 normal)
        {
            float2 uv = rng.NextFloat2();

            // uniform sampling of a hemisphere
            // from : https://cg.informatik.uni-freiburg.de/course_notes/graphics2_08_renderingEquation.pdf (inversion method, page 42)
            float u = uv.x;
            float radius = sqrt(2 * u - u * u);
            float theta = uv.y * 2 * PI;
            sincos(theta, out float sinTheta, out float cosTheta);
            float2 xz = radius * float2(cosTheta, sinTheta);
            float3 tangentSpaceDirection = float3(xz.x, 1 - u, xz.y);

            return TangentToWorldSpace(tangentSpaceDirection, normal);
        }

        public static float3 OnCosineWeightedHemisphere(this ref Random rng, float3 normal)
        {
            float2 uv = rng.NextFloat2();

            // uniform sampling of a cosine-weighted hemisphere
            // from : https://cg.informatik.uni-freiburg.de/course_notes/graphics2_08_renderingEquation.pdf (inversion method, page 47)
            // same algorithm used here : http://www.rorydriscoll.com/2009/01/07/better-sampling/
            float u = uv.x;
            float radius = sqrt(u);
            float theta = uv.y * 2 * PI;
            sincos(theta, out float sinTheta, out float cosTheta);
            float2 xz = radius * float2(cosTheta, sinTheta);
            float3 tangentSpaceDirection = float3(xz.x, sqrt(1 - u), xz.y);

            return TangentToWorldSpace(tangentSpaceDirection, normal);
        }

        public static float3 TangentToWorldSpace(float3 tangentSpaceVector, float3 normal)
        {
            GetOrthonormalBasis(normal, out float3 tangent, out float3 bitangent);

            float3x3 orthogonalMatrix = float3x3(tangent, normal, bitangent);
            float3 result = mul(orthogonalMatrix, tangentSpaceVector);
            return normalize(result);
        }

        public static float3 WorldToTangentSpace(float3 worldSpaceVector, float3 normal)
        {
            GetOrthonormalBasis(normal, out float3 tangent, out float3 bitangent);

            float3x3 orthogonalMatrix = float3x3(tangent, normal, bitangent);
            float3 result = mul(transpose(orthogonalMatrix), worldSpaceVector);
            return normalize(result);
        }

        public static float3 SphericalToCartesian(float theta, float phi)
        {
            sincos(float2(theta, phi), out float2 sinThetaPhi, out float2 cosThetaPhi);
            return float3(sinThetaPhi.x * cosThetaPhi.y, cosThetaPhi.x, sinThetaPhi.x * sinThetaPhi.y);
        }

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