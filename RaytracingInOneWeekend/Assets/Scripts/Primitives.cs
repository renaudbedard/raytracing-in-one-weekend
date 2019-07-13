using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Assertions;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
#if SOA_SPHERES
    struct SoaSpheres : IDisposable
    {
        public NativeArray<float3> Center;
        public NativeArray<float> Radius;
        public NativeArray<ushort> MaterialIndex;

        public int Count => Center.Length;

        public void Dispose()
        {
            if (Center.IsCreated) Center.Dispose();
            if (Radius.IsCreated) Radius.Dispose();
            if (MaterialIndex.IsCreated) MaterialIndex.Dispose();
        }
    }
#else
    enum PrimitiveType
    {
        None,
        Sphere
    }

    struct Primitive
    {
        public readonly PrimitiveType Type;

        [ReadOnly] readonly NativeSlice<Sphere> sphere;

        // TODO: do we need a public accessor to the underlying primitive?

        public Primitive(NativeSlice<Sphere> sphere)
        {
            Assert.IsTrue(sphere.Length == 1, "Primitive cannot be multi-valued");
            Type = PrimitiveType.Sphere;
            this.sphere = sphere;
        }

        public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
        {
            switch (Type)
            {
                case PrimitiveType.Sphere:
                    return sphere[0].Hit(r, tMin, tMax, out rec);

                default:
                    rec = default;
                    return false;
            }
        }
    }

    struct Sphere
    {
        public readonly float3 Center;
        public readonly float Radius;
        public readonly Material Material;

        public Sphere(float3 center, float radius, Material material)
        {
            Center = center;
            Radius = radius;
            Material = material;
        }
    }
#endif

    static class WorldExtensions
    {
#if SOA_SPHERES
        public static bool Hit(this SoaSpheres spheres, Ray r, float tMin, float tMax, out HitRecord rec)
        {
            bool hitAnything = false;
            rec = new HitRecord(tMax, 0, 0, default);

            for (var i = 0; i < spheres.Count; i++)
            {
                // TODO: 4-wide hit
                if (spheres.Hit(i, r, tMin, rec.Distance, out HitRecord thisRec))
                {
                    hitAnything = true;
                    rec = thisRec;
                }
            }

            return hitAnything;
        }
#else
        public static bool Hit(this NativeArray<Primitive> primitives, Ray r, float tMin, float tMax, out HitRecord rec)
        {
            bool hitAnything = false;
            rec = new HitRecord(tMax, 0, 0, default);

            for (var i = 0; i < primitives.Length; i++)
            {
                Primitive primitive = primitives[i];
                if (primitive.Hit(r, tMin, rec.Distance, out HitRecord thisRec))
                {
                    hitAnything = true;
                    rec = thisRec;
                }
            }

            return hitAnything;
        }
#endif

#if SOA_SPHERES
        private static bool Hit(this SoaSpheres spheres, int sphereIndex, Ray r, float tMin, float tMax, out HitRecord rec)
#else
        public static bool Hit(this Sphere s, Ray r, float tMin, float tMax, out HitRecord rec)
#endif
        {
#if SOA_SPHERES
            float3 center = spheres.Center[sphereIndex];
            float radius = spheres.Radius[sphereIndex];
            ushort material = spheres.MaterialIndex[sphereIndex];
#else
            float3 center = s.Center;
            float radius = s.Radius;
            Material material = s.Material;
#endif

            float3 oc = r.Origin - center;
            float a = dot(r.Direction, r.Direction);
            float b = dot(oc, r.Direction);
            float c = dot(oc, oc) - radius * radius;
            float discriminant = b * b - a * c;

            if (discriminant > 0)
            {
                float sqrtDiscriminant = sqrt(discriminant);
                float t = (-b - sqrtDiscriminant) / a;
                if (t < tMax && t > tMin)
                {
                    float3 point = r.GetPoint(t);
                    rec = new HitRecord(t, point, (point - center) / radius, material);
                    return true;
                }

                t = (-b + sqrtDiscriminant) / a;
                if (t < tMax && t > tMin)
                {
                    float3 point = r.GetPoint(t);
                    rec = new HitRecord(t, point, (point - center) / radius, material);
                    return true;
                }
            }

            rec = default;
            return false;
        }
    }
}