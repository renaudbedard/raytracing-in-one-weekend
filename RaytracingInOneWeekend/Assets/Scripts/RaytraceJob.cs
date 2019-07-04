using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
    [BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
    struct RaytraceJob : IJobParallelFor
    {
        [ReadOnly] public int2 Size;
        [ReadOnly] public Camera Camera;
        [ReadOnly] public NativeArray<Sphere> Spheres;
        [ReadOnly] public int SampleCount;
        [ReadOnly] public int TraceDepth;
        [ReadOnly] public Random Rng;

        [WriteOnly] public NativeArray<half4> Target;

        bool Color(Ray r, int depth, out float3 color)
        {
            if (Spheres.Hit(r, 0.001f, float.PositiveInfinity, out HitRecord rec))
            {
                if (depth < TraceDepth && rec.Material.Scatter(r, rec, Rng, out float3 attenuation, out Ray scattered))
                {
                    if (Color(scattered, depth + 1, out float3 scatteredColor))
                    {
                        color = attenuation * scatteredColor;
                        return true;
                    }
                }
                color = default;
                return false;
            }

            float3 unitDirection = normalize(r.Direction);
            float t = 0.5f * (unitDirection.y + 1);
            color = lerp(1, float3(0.5f, 0.7f, 1), t);
            return true;
        }

        public void Execute(int index)
        {
            int2 coordinates = int2(
                index % Size.x, // column 
                index / Size.x // row
            );

            float3 colorAcc = 0;
            int realSampleCount = 0;
            for (int s = 0; s < SampleCount; s++)
            {
                float2 normalizedCoordinates = (coordinates + Rng.NextFloat2()) / Size; // (u, v)
                Ray r = Camera.GetRay(normalizedCoordinates);
                if (Color(r, 0, out float3 sampleColor))
                {
                    colorAcc += sampleColor;
                    realSampleCount++;
                }
            }

            float3 finalColor;
            if (realSampleCount == 0)
                finalColor = float3(1, 0, 1);
            else
            {
                finalColor = colorAcc / realSampleCount;
                finalColor = sqrt(finalColor);
            }

            Target[index] = half4(half3(finalColor), half(1));
        }
    }
}