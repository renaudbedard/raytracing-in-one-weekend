using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;

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

        public static float3 LinearToGamma(this float3 value)
        {
            value = max(value, 0);
            return max(1.055f * pow(value, 0.416666667f) - 0.055f, 0);
        }
    }
}