using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
    enum PrimitiveType
    {
        None,
        Sphere
    }

    unsafe struct Primitive : IPrimitive
    {
        public readonly PrimitiveType Type;

        readonly Sphere* sphere;

        // TODO: do we need a public accessor to the underlying one?

        public Primitive(Sphere* sphere)
        {
            Type = PrimitiveType.Sphere;
            this.sphere = sphere;
        }

        public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
        {
            switch (Type)
            {
                case PrimitiveType.Sphere:
                    return sphere->Hit(r, tMin, tMax, out rec);

                default:
                    rec = default;
                    return false;
            }
        }
    }

    static class PrimitiveExtensions
    {
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
    }

    interface IPrimitive
    {
        bool Hit(Ray r, float tMin, float tMax, out HitRecord rec);
    }

    struct Sphere : IPrimitive
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

        public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
        {
            float3 oc = r.Origin - Center;
            float a = dot(r.Direction, r.Direction);
            float b = dot(oc, r.Direction);
            float c = dot(oc, oc) - Radius * Radius;
            float discriminant = b * b - a * c;

            if (discriminant > 0)
            {
                float sqrtDiscriminant = sqrt(discriminant);
                float t = (-b - sqrtDiscriminant) / a;
                if (t < tMax && t > tMin)
                {
                    float3 point = r.GetPoint(t);
                    rec = new HitRecord(t, point, (point - Center) / Radius, Material);
                    return true;
                }

                t = (-b + sqrtDiscriminant) / a;
                if (t < tMax && t > tMin)
                {
                    float3 point = r.GetPoint(t);
                    rec = new HitRecord(t, point, (point - Center) / Radius, Material);
                    return true;
                }
            }

            rec = default;
            return false;
        }
    }
}