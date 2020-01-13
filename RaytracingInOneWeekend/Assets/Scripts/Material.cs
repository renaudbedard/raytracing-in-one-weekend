using System;
using JetBrains.Annotations;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using static RaytracerInOneWeekend.MathExtensions;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	enum MaterialType
	{
		None,
		Lambertian,
		Metal,
		Dielectric,
		DiffuseLight,
		ProbabilisticVolume
	}

	struct Material
	{
		public readonly MaterialType Type;
		public readonly Texture Texture;
		public readonly float2 TextureScale;
		public readonly float Parameter;

		float Roughness => Parameter; // for Metal
		float RefractiveIndex => Parameter; // for Dielectric
		public float Density => Parameter; // for ProbabilisticVolume

		public Material(MaterialType type, float2 textureScale, Texture albedo = default,
			Texture emission = default, float roughness = 0, float refractiveIndex = 1, float density = 1) : this()
		{
			Type = type;
			TextureScale = textureScale;

			switch (type)
			{
				case MaterialType.Lambertian: Texture = albedo; break;
				case MaterialType.Metal: Parameter = saturate(roughness); Texture = albedo; break;
				case MaterialType.Dielectric: Parameter = refractiveIndex; break;
				case MaterialType.DiffuseLight: Texture = emission; break;
				case MaterialType.ProbabilisticVolume: Parameter = density; Texture = albedo; break;
			}
		}

		[Pure]
		public bool Scatter(Ray ray, HitRecord rec, ref Random rng, PerlinData perlinData,
			out float3 reflectance, out Ray scattered)
		{
			switch (Type)
			{
				case MaterialType.Lambertian:
				{
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					float3 randomDirection = rng.OnCosineWeightedHemisphere(rec.Normal);
					//float3 randomDirection = rng.OnUniformHemisphere(rec.Normal);
					scattered = new Ray(rec.Point, randomDirection, ray.Time);
					return true;
				}

				case MaterialType.Metal:
				{
					// GGX (probably wrong though)
					// float3 outgoingDirection = WorldToTangentSpace(-ray.Direction, rec.Normal);
					// if (GgxMicrofacet.ImportanceSample(Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData),
					// 	Roughness, ref rng, outgoingDirection, out float3 toLight, out reflectance))
					// {
					// 	float3 scatterDirection = TangentToWorldSpace(toLight, rec.Normal);
					// 	scattered = new Ray(rec.Point, scatterDirection, ray.Time);
					// 	return true;
					// }
					// scattered = default;
					// return false;

					// Peter Shirley's fuzzy metal
					float3 reflected = reflect(ray.Direction, rec.Normal);
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					scattered = new Ray(rec.Point, normalize(reflected + Roughness * rng.NextFloat3Direction()), ray.Time);
					return true;
				}

				case MaterialType.Dielectric:
				{
					float3 reflected = reflect(ray.Direction, rec.Normal);
					reflectance = 1;
					float niOverNt;
					float3 outwardNormal;
					float cosine;

					if (dot(ray.Direction, rec.Normal) > 0)
					{
						outwardNormal = -rec.Normal;
						niOverNt = RefractiveIndex;
						cosine = RefractiveIndex * dot(ray.Direction, rec.Normal);
					}
					else
					{
						outwardNormal = rec.Normal;
						niOverNt = 1 / RefractiveIndex;
						cosine = -dot(ray.Direction, rec.Normal);
					}

					if (Refract(ray.Direction, outwardNormal, niOverNt, out float3 refracted))
					{
						float reflectProb = Schlick(cosine, RefractiveIndex);
						scattered = new Ray(rec.Point, rng.NextFloat() < reflectProb ? reflected : refracted, ray.Time);
					}
					else
						scattered = new Ray(rec.Point, reflected, ray.Time);

					return true;
				}

				case MaterialType.ProbabilisticVolume:
					scattered = new Ray(rec.Point, rng.NextFloat3Direction());
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					return true;

				default:
					reflectance = default;
					scattered = default;
					return false;
			}
		}

		[Pure]
		public float Pdf(float3 incomingLightDirection, float3 outgoingLightDirection, float3 geometricNormal)
		{
			switch (Type)
			{
				case MaterialType.Lambertian:
					return max(dot(geometricNormal, incomingLightDirection) / PI, 0);

				case MaterialType.ProbabilisticVolume:
					return 1.0f / (4.0f * PI);

				case MaterialType.Metal:
					throw new NotImplementedException();
					// Disabled because it current does not work right
					//return GgxMicrofacet.Pdf(incomingLightDirection, outgoingLightDirection, geometricNormal, Roughness);

				default: throw new NotImplementedException();
			}
		}

		[Pure]
		public float3 Emit(float3 position, float3 normal, PerlinData perlinData)
		{
			switch (Type)
			{
				case MaterialType.DiffuseLight: return Texture.Value(position, normal, TextureScale, perlinData);
				default: return 0;
			}
		}

		public bool IsPerfectSpecular
		{
			get
			{
				switch (Type)
				{
					case MaterialType.Dielectric: return true;
					case MaterialType.Metal: return Roughness.AlmostEquals(0);
				}
				return false;
			}
		}

		static bool Refract(float3 v, float3 n, float niOverNt, out float3 refracted)
		{
			float dt = dot(v, n);
			float discriminant = 1 - niOverNt * niOverNt * (1 - dt * dt);
			if (discriminant > 0)
			{
				refracted = niOverNt * (v - n * dt) - n * sqrt(discriminant);
				return true;
			}

			refracted = default;
			return false;
		}

		static float Schlick(float radians, float refractiveIndex)
		{
			float r0 = (1 - refractiveIndex) / (1 + refractiveIndex);
			r0 *= r0;
			float exponential = pow(1 - radians, 5);
			return r0 + (1 - r0) * exponential;
		}
	}
}