using System;
using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
#if SOA_SPHERES
    struct SoaMaterials : IDisposable
    {
        public NativeArray<float3> Albedo;
        public NativeArray<byte> Type;
        public NativeArray<float> Parameter; // Fuzz for Metal, RefractiveIndex for Dielectric, otherwise unused

        public int Count => Albedo.Length; 
        
        public void Dispose()
        {
            if (Albedo.IsCreated) Albedo.Dispose();
            if (Type.IsCreated) Type.Dispose();
            if (Parameter.IsCreated) Parameter.Dispose();
        }
    }
#else
    struct Material
    {
        public readonly MaterialType Type;
        public readonly float3 Albedo;
        public readonly float Fuzz;
        public readonly float RefractiveIndex;

        public Material(MaterialType type, float3 albedo = default, float fuzz = 0, float refractiveIndex = 1)
        {
            Type = type;
            Albedo = albedo;
            Fuzz = saturate(fuzz);
            RefractiveIndex = refractiveIndex;
        }
    }
#endif

    enum MaterialType : byte
    {
        None,
        Lambertian,
        Metal,
        Dielectric
    }

    static class MaterialExtensions
    {
#if SOA_SPHERES
        public static bool Scatter(this SoaMaterials materials, Ray r, HitRecord rec, Random rng, out float3 attenuation, out Ray scattered)
#else
        public static bool Scatter(this Material m, Ray r, HitRecord rec, Random rng, out float3 attenuation, out Ray scattered)
#endif
        {
#if SOA_SPHERES
            var type = (MaterialType) materials.Type[rec.MaterialIndex];
            float3 albedo = materials.Albedo[rec.MaterialIndex];
            float fuzz = materials.Parameter[rec.MaterialIndex];
            float refractiveIndex = materials.Parameter[rec.MaterialIndex];
#else
            MaterialType type = m.Type;
            float3 albedo = m.Albedo;
            float fuzz = m.Fuzz;
            float refractiveIndex = m.RefractiveIndex;
#endif

            switch (type)
            {
                case MaterialType.Lambertian:
                {
                    float3 target = rec.Point + rec.Normal + rng.UnitVector();
                    scattered = new Ray(rec.Point, target - rec.Point);
                    attenuation = albedo;
                    return true;
                }

                case MaterialType.Metal:
                {
                    float3 reflected = reflect(normalize(r.Direction), rec.Normal);
                    scattered = new Ray(rec.Point, reflected + fuzz * rng.InUnitSphere());
                    attenuation = albedo;
                    return dot(scattered.Direction, rec.Normal) > 0;
                }

                case MaterialType.Dielectric:
                {
                    float3 reflected = reflect(r.Direction, rec.Normal);
                    attenuation = 1;
                    float niOverNt;
                    float3 outwardNormal;
                    float cosine;

                    if (dot(r.Direction, rec.Normal) > 0)
                    {
                        outwardNormal = -rec.Normal;
                        niOverNt = refractiveIndex;
                        cosine = refractiveIndex * dot(r.Direction, rec.Normal) / length(r.Direction);
                    }
                    else
                    {
                        outwardNormal = rec.Normal;
                        niOverNt = 1 / refractiveIndex;
                        cosine = -dot(r.Direction, rec.Normal) / length(r.Direction);
                    }

                    if (Refract(r.Direction, outwardNormal, niOverNt, out float3 refracted))
                    {
                        float reflectProb = Schlick(cosine, refractiveIndex);
                        scattered = new Ray(rec.Point, rng.NextFloat() < reflectProb ? reflected : refracted);
                    }
                    else
                        scattered = new Ray(rec.Point, reflected);

                    return true;
                }

                default:
                    attenuation = default;
                    scattered = default;
                    return false;
            }
        }

        static bool Refract(float3 v, float3 n, float niOverNt, out float3 refracted)
        {
            float3 normalizedV = normalize(v);
            float dt = dot(normalizedV, n);
            float discriminant = 1 - niOverNt * niOverNt * (1 - dt * dt);
            if (discriminant > 0)
            {
                refracted = niOverNt * (normalizedV - n * dt) - n * sqrt(discriminant);
                return true;
            }

            refracted = default;
            return false;
        }

        static float Schlick(float cosine, float refractiveIndex)
        {
            float r0 = (1 - refractiveIndex) / (1 + refractiveIndex);
            r0 *= r0;
            return r0 + (1 - r0) * pow((1 - cosine), 5);
        }
    }
}