using System.Runtime.CompilerServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
    static class MathExtensions
    {
        // TODO : stratified sampling

        public static float2 InUnitDisk(this Random rng)
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
            // Hugues-Möller method
            // from : https://expf.wordpress.com/2010/05/05/building-an-orthonormal-basis-from-a-unit-vector/

            float3 absNormal = abs(normal);

            // TODO: there's probably a more clever and less branching way to do this
            int minComponent = 0;
            float minValue = absNormal.x;
            for (int i = 1; i < 3; i++)
            {
                if (absNormal[i] < minValue)
                {
                    minValue = absNormal[i];
                    minComponent = i;
                }
            }
            switch (minComponent)
            {
                case 0: tangent = float3(0, -normal.z, normal.y); break;
                case 1: tangent = float3(-normal.z, 0, normal.x); break;
                //case 2:
                default:
                    tangent = float3(-normal.y, normal.x, 0);
                    break;
            }

            tangent = normalize(tangent);
            bitangent = cross(normal, tangent);

            // ----------

            // Corrected Frisvad method
            // from listing 3 in : https://graphics.pixar.com/library/OrthonormalB/paper.pdf
            //
            // float s = normal.z >= 0 ? 1.0f : -1.0f;
            // float a = -1 / (s + normal.z);
            // float b = normal.x * normal.y * a;
            // tangent = float3(1 + s * normal.x * normal.x * a, s * b, -s * normal.x);
            // bitangent = float3(b, s + normal.y * normal.y * a, -normal.y);

            // ----------

            // Combined method
            // from listing 2 in : http://jcgt.org/published/0006/01/02/paper.pdf
            //
            // const double dThreshold = -0.9999999999776;
            // const float rThreshold = -0.7f;
            // if (normal.z >= rThreshold)
            // {
            //     float a = 1 / (1 + normal.z);
            //     float b = -normal.x * normal.y * a;
            //     tangent = float3(1 - normal.x * normal.x * a, b, -normal.x);
            //     bitangent = float3(b, 1 - normal.y * normal.y * a, -normal.y);
            // }
            // else
            // {
            //     double3 normalD = normal;
            //     double d = 1 / sqrt(normalD.x * normalD.x + normalD.y * normalD.y + normalD.z * normalD.z);
            //     normalD *= d;
            //     if(normalD.z >= dThreshold)
            //     {
            //         double a = 1 / (1 + normalD.z);
            //         double b = -normalD.x * normalD.y * a;
            //         tangent = float3(1 - (float) (normalD.x * normalD.x * a), (float) b, (float) -normalD.x);
            //         bitangent = float3((float) b, 1 - (float) (normalD.y * normalD.y * a), (float) -normalD.y);
            //     }
            //     else
            //     {
            //         tangent = float3(0, -1, 0);
            //         bitangent = float3(-1, 0, 0);
            //     }
            // }
        }

        public static float3 OnUniformHemisphere(this Random rng, float3 normal)
        {
            // uniform sampling of a hemisphere
            // from : https://cg.informatik.uni-freiburg.de/course_notes/graphics2_08_renderingEquation.pdf (inversion method, page 42)
            float u = rng.NextFloat();
            float radius = sqrt(2 * u - u * u);
            float theta = rng.NextFloat(0, 2 * PI);
            sincos(theta, out float sinTheta, out float cosTheta);
            float3 tangentSpaceDirection = float3(radius * float2(cosTheta, sinTheta), 1 - u);

            // build an orthonormal basis from the forward (normal) vector
            GetOrthonormalBasis(normal, out float3 tangent, out float3 bitangent);

            // transform from tangent-space to world-space
            float3x3 rotationMatrix = float3x3(tangent, bitangent, normal);
            float3 result = mul(rotationMatrix, tangentSpaceDirection);
            return normalize(result);
        }

        public static float3 OnCosineWeightedHemisphere(this Random rng, float3 normal)
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
            GetOrthonormalBasis(normal, out float3 tangent, out float3 bitangent);

            // transform from tangent-space to world-space
            float3x3 rotationMatrix = float3x3(tangent, bitangent, normal);
            return mul(rotationMatrix, tangentSpaceDirection);
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
            return abs(rhs - lhs) < 1e-7f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool2 AlmostEquals(this float2 lhs, float2 rhs)
        {
            return abs(rhs - lhs) < 1e-7f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3 AlmostEquals(this float3 lhs, float3 rhs)
        {
            return abs(rhs - lhs) < 1e-7f;
        }
    }
}