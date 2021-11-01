using System;
using JetBrains.Annotations;
using Unity.Mathematics;
using Util;
using static Unity.Mathematics.math;

namespace Runtime
{
	enum MaterialType
	{
		Standard,
		Dielectric,
		ProbabilisticVolume
	}

	readonly struct Material
	{
		public readonly MaterialType Type;
		public readonly Texture Albedo, Glossiness, Emission, Metallic;

		readonly float parameter;
		public float RefractiveIndex => parameter; // for Dielectric
		public float Density => parameter; // for ProbabilisticVolume

		public Material(MaterialType type, Texture albedo = default, Texture emission = default, Texture glossiness = default, Texture metallic = default, float refractiveIndex = 1.46f, float density = 1) : this()
		{
			Type = type;
			Albedo = albedo;
			Emission = emission;
			Glossiness = glossiness;
			Metallic = metallic;

			switch (type)
			{
				case MaterialType.Dielectric: parameter = refractiveIndex; break;
				case MaterialType.ProbabilisticVolume: parameter = density; break;
			}
		}

		[Pure]
		public bool ProbabilisticHit(ref float hitDistance, ref RandomSource rng)
		{
			if (Type != MaterialType.ProbabilisticVolume)
				return false;

			float volumeHitDistance = -(1 / max(Density, EPSILON)) * log(rng.NextFloat());

			if (volumeHitDistance < hitDistance)
			{
				hitDistance = volumeHitDistance;
				return true;
			}

			return false;
		}

		[Pure]
		public void Scatter(Ray ray, HitRecord rec, ref RandomSource rng, PerlinNoise perlinNoise,
			out float3 reflectance, out Ray scattered)
		{
			reflectance = Albedo.SampleColor(rec.TexCoords, perlinNoise);

			switch (Type)
			{
				case MaterialType.Standard:
				{
					float metallicChance = Metallic.SampleScalar(rec.TexCoords, perlinNoise);
					if (rng.NextFloat() < metallicChance)
					{
						// Peter Shirley's fuzzy metal
						float3 reflected = reflect(ray.Direction, rec.Normal);
						float roughness = 1 - Glossiness.SampleScalar(rec.TexCoords, perlinNoise);
						scattered = new Ray(rec.Point, normalize(reflected + roughness * rng.NextFloat3Direction()), ray.Time);
					}
					else
					{
						// Rough Plastic
						float roughness = 1 - Glossiness.SampleScalar(rec.TexCoords, perlinNoise);
						float3 roughNormal = normalize(rec.Normal + roughness * rng.NextFloat3Direction());
						float cosine;
						if (dot(ray.Direction, roughNormal) > 0)
							cosine = RefractiveIndex * dot(ray.Direction, roughNormal);
						else
							cosine = -dot(ray.Direction, roughNormal);

						// TODO: Using this Fresnel equation might not make sense for conductors (or plastics)
						if (rng.NextFloat() < Schlick(cosine, RefractiveIndex))
						{
							// Lambertian diffuse
							float3 randomDirection = rng.OnUniformHemisphere(rec.Normal); // TODO: Is cosine hemisphere more accurate?
							scattered = new Ray(rec.Point, randomDirection, ray.Time);
						}
						else
						{
							// Glossy reflection
							scattered = new Ray(rec.Point, reflect(ray.Direction, roughNormal), ray.Time);
						}
					}
					break;
				}

				case MaterialType.Dielectric:
				{
					float roughness = 1 - Glossiness.SampleScalar(rec.TexCoords, perlinNoise);
					float3 roughNormal = normalize(rec.Normal + roughness * rng.NextFloat3Direction());

					float niOverNt, cosine;
					float3 outwardRoughNormal;
					if (dot(ray.Direction, roughNormal) > 0)
					{
						outwardRoughNormal = -roughNormal;
						niOverNt = RefractiveIndex;
						cosine = RefractiveIndex * dot(ray.Direction, roughNormal);
					}
					else
					{
						outwardRoughNormal = roughNormal;
						niOverNt = 1 / RefractiveIndex;
						cosine = -dot(ray.Direction, roughNormal);
					}

					float3 scatterDirection;
					if (Refract(ray.Direction, outwardRoughNormal, niOverNt, out float3 refracted) &&
					    rng.NextFloat() > Schlick(cosine, RefractiveIndex))
					{
						scatterDirection = refracted;
					}
					else
						scatterDirection = reflect(ray.Direction, roughNormal);

					scattered = new Ray(rec.Point, scatterDirection, ray.Time);
					break;
				}

				case MaterialType.ProbabilisticVolume:
					scattered = new Ray(rec.Point, rng.NextFloat3Direction());
					break;

				default:
					throw new NotImplementedException();
			}
		}

		[Pure]
		public float3 Emit(float2 texCoords, PerlinNoise perlinNoise)
		{
			return Emission.SampleColor(texCoords, perlinNoise);
		}

		public bool IsPerfectSpecular
		{
			get
			{
				switch (Type)
				{
					case MaterialType.Dielectric:
						return true;

					case MaterialType.Standard:
						return Metallic.Type == TextureType.Constant && all(Metallic.MainColor.AlmostEquals(1)) &&
						       Glossiness.Type == TextureType.Constant && all(Glossiness.MainColor.AlmostEquals(1));
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