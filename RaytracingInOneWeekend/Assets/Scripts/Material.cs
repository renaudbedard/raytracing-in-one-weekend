using JetBrains.Annotations;
using Sirenix.OdinInspector;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	enum MaterialType
	{
		None,
		Lambertian,
		Metal,
		Dielectric,
		DiffuseLight
	}

	struct Material
	{
		public readonly MaterialType Type;
		public readonly Texture Texture;
		public readonly float2 TextureScale;
		public readonly float Parameter;

		public Material(MaterialType type, float2 textureScale, Texture albedo = default,
			Texture emission = default, float fuzz = 0, float refractiveIndex = 1) : this()
		{
			Type = type;
			TextureScale = textureScale;

			switch (type)
			{
				case MaterialType.Lambertian: Texture = albedo; break;
				case MaterialType.Metal: Parameter = saturate(fuzz); Texture = albedo; break;
				case MaterialType.Dielectric: Parameter = refractiveIndex; break;
				case MaterialType.DiffuseLight: Texture = emission; break;
			}
		}

		[Pure]
		public bool Scatter(Ray r, HitRecord rec, Random rng, PerlinData perlinData,
			out float3 attenuation, out Ray scattered)
		{
			switch (Type)
			{
				case MaterialType.Lambertian:
				{
					attenuation = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					scattered = new Ray(rec.Point, normalize(rec.Normal + rng.NextFloat3Direction()), r.Time);

					// NOTE: both of those alternate methods fail with artifacts for reasons unknown
					//scattered = new Ray(rec.Point, rng.OnUniformHemisphere(rec.Normal), r.Time);
					//scattered = new Ray(rec.Point, rng.OnCosineWeightedHemisphere(rec.Normal), r.Time);
					return true;
				}

				case MaterialType.Metal:
				{
					float fuzz = Parameter;
					float3 reflected = reflect(r.Direction, rec.Normal);
					attenuation = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					scattered = new Ray(rec.Point, normalize(reflected + fuzz * rng.NextFloat3Direction()), r.Time);
					return true;
				}

				case MaterialType.Dielectric:
				{
					float refractiveIndex = Parameter;
					float3 reflected = reflect(r.Direction, rec.Normal);
					attenuation = 1;
					float niOverNt;
					float3 outwardNormal;
					float cosine;

					if (dot(r.Direction, rec.Normal) > 0)
					{
						outwardNormal = -rec.Normal;
						niOverNt = refractiveIndex;
						cosine = refractiveIndex * dot(r.Direction, rec.Normal);
					}
					else
					{
						outwardNormal = rec.Normal;
						niOverNt = 1 / refractiveIndex;
						cosine = -dot(r.Direction, rec.Normal);
					}

					if (Refract(r.Direction, outwardNormal, niOverNt, out float3 refracted))
					{
						float reflectProb = Schlick(cosine, refractiveIndex);
						scattered = new Ray(rec.Point, rng.NextFloat() < reflectProb ? reflected : refracted, r.Time);
					}
					else
						scattered = new Ray(rec.Point, reflected, r.Time);

					return true;
				}

				default:
					attenuation = default;
					scattered = default;
					return false;
			}
		}

		[Pure]
		public float3 Emit(float3 position, float3 normal, PerlinData perlinData)
		{
			switch (Type)
			{
				case MaterialType.DiffuseLight:
					return Texture.Value(position, normal, TextureScale, perlinData);

				default:
					return 0;
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