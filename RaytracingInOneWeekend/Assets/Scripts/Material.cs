using JetBrains.Annotations;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
    enum MaterialType : byte
    {
        None,
        Lambertian,
        Metal,
        Dielectric
    }

    struct Material
    {
        public readonly MaterialType Type;
        public readonly float3 Albedo;
        public readonly float Fuzz;
        public readonly float RefractiveIndex;

        public Material(MaterialType type, float3 albedo = default, float fuzz = 0, float refractiveIndex = 0)
        {
            Type = type;
            Albedo = albedo;
            Fuzz = saturate(fuzz);
            RefractiveIndex = refractiveIndex;
        }
        
        public static Material Lambertian(float3 albedo) => new Material(MaterialType.Lambertian, albedo);
        public static Material Metal(float3 albedo, float fuzz = 0) => new Material(MaterialType.Metal, albedo, fuzz);
        public static Material Dielectric(float refractiveIndex) => new Material(MaterialType.Dielectric, refractiveIndex: refractiveIndex);

        [Pure]
        public bool Scatter(Ray r, HitRecord rec, Random rng, out float3 attenuation, out Ray scattered)
        {
            switch (Type)
            {
                case MaterialType.Lambertian:
                {
                    float3 target = rec.Point + rec.Normal + rng.UnitVector();
                    scattered = new Ray(rec.Point, target - rec.Point);
                    attenuation = Albedo;
                    return true;
                }

                case MaterialType.Metal:
                {
                    float3 reflected = reflect(normalize(r.Direction), rec.Normal);
                    scattered = new Ray(rec.Point, reflected + Fuzz * rng.InUnitSphere());
                    attenuation = Albedo;
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
                        niOverNt = RefractiveIndex;
                        cosine = RefractiveIndex * dot(r.Direction, rec.Normal) / length(r.Direction);
                    }
                    else
                    {
                        outwardNormal = rec.Normal;
                        niOverNt = 1 / RefractiveIndex;
                        cosine = -dot(r.Direction, rec.Normal) / length(r.Direction);
                    }

                    if (Refract(r.Direction, outwardNormal, niOverNt, out float3 refracted))
                    {
                        float reflectProb = Schlick(cosine, RefractiveIndex);
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