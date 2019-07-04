using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
    static class Extensions
    {
        public static bool Hit(this NativeArray<Sphere> spheres, Ray r, float tMin, float tMax, out HitRecord rec)
        {
            bool hitAnything = false;
            rec = new HitRecord(tMax, 0, 0, default);

            for (var i = 0; i < spheres.Length; i++)
            {
                Sphere sphere = spheres[i];
                if (sphere.Hit(r, tMin, rec.Distance, out HitRecord thisRec))
                {
                    hitAnything = true;
                    rec = thisRec;
                }
            }

            return hitAnything;
        }

        public static float3 InUnitSphere(this Random rng)
        {
            // TODO: is this really as fast it gets?
            float3 p;
            do
            {
                p = 2 * rng.NextFloat3() - 1;
            } while (lengthsq(p) >= 1);

            return p;
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
    }
}