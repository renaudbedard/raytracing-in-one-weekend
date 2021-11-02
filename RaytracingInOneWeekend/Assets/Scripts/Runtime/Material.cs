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
		public float IndexOfRefraction => parameter; // for Dielectric and Standard
		public float Density => parameter; // for ProbabilisticVolume

		public Material(MaterialType type, Texture albedo = default, Texture emission = default, Texture glossiness = default, Texture metallic = default, float indexOfRefraction = 1.46f, float density = 1) : this()
		{
			Type = type;
			Albedo = albedo;
			Emission = emission;
			Glossiness = glossiness;
			Metallic = metallic;

			switch (type)
			{
				case MaterialType.Dielectric: parameter = indexOfRefraction; break;
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
						float eta = 1 / IndexOfRefraction;

						// Rough Plastic
						float roughness = 1 - Glossiness.SampleScalar(rec.TexCoords, perlinNoise);
						float3 roughNormal = normalize(rec.Normal + roughness * rng.NextFloat3Direction());
						float cosTheta = min(dot(-ray.Direction, roughNormal), 1);

						// TODO: Using this Fresnel equation might not make sense for conductors (or plastics)
						if (rng.NextFloat() < Schlick(cosTheta, IndexOfRefraction))
						{
							// Lambertian diffuse
							float3 randomDirection = rng.OnCosineWeightedHemisphere(rec.Normal);
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
						niOverNt = IndexOfRefraction;
						cosine = IndexOfRefraction * dot(ray.Direction, roughNormal);
					}
					else
					{
						outwardRoughNormal = roughNormal;
						niOverNt = 1 / IndexOfRefraction;
						cosine = -dot(ray.Direction, roughNormal);
					}

					float3 scatterDirection;
					if (Refract(ray.Direction, outwardRoughNormal, niOverNt, out float3 refracted) &&
					    rng.NextFloat() > Schlick(cosine, IndexOfRefraction))
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

		static float Schlick(float cosine, float refractiveIndex)
		{
			float r0 = (1 - refractiveIndex) / (1 + refractiveIndex);
			r0 *= r0;
			return r0 + (1 - r0) * pow(1 - cosine, 5);
		}

		// From : https://github.com/yblein/tracing/blob/master/src/material.rs#L648
		static float PlasticFresnel(float eta, float cos_i)
		{
			// clamp cos_i before using trigonometric identities
			cos_i = clamp(cos_i, -1, 1);

			float sin_t2 = eta * eta * (1 - cos_i * cos_i);
			if (sin_t2 > 1.0)
			{
				// Total Internal Reflection
				return 1;
			}

			float cos_t = sqrt(1 - sin_t2);
			float r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
			float r_p = (eta * cos_t - cos_i) / (eta * cos_t + cos_i);
			return (r_s * r_s + r_p * r_p) * 0.5f;
		}
	}
}