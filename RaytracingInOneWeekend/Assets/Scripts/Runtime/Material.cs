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

		public Material(MaterialType type, Texture albedo = default, Texture emission = default, Texture glossiness = default, Texture metallic = default, float indexOfRefraction = 1.5f, float density = 1) : this()
		{
			Type = type;
			Albedo = albedo;
			Emission = emission;
			Glossiness = glossiness;
			Metallic = metallic;

			switch (type)
			{
				case MaterialType.Dielectric:
				case MaterialType.Standard:
					parameter = indexOfRefraction;
					break;

				case MaterialType.ProbabilisticVolume:
					parameter = density;
					break;
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
					float roughness = 1 - Glossiness.SampleScalar(rec.TexCoords, perlinNoise);
					float3 roughNormal = normalize(rec.Normal + roughness * rng.NextFloat3Direction() * 0.5f);
					float incidentCosine = -dot(ray.Direction, rec.Normal); // TODO: Should this use the rough normal instead?

					if (rng.NextFloat() < metallicChance)
					{
						// Rough metal
						scattered = new Ray(rec.Point, reflect(ray.Direction, roughNormal), ray.Time);

						// TODO: Cook-Torrance BRDF
						//float maskingShadowing = Microfacet.SmithMaskingShadowing(ray.Direction, roughness, rec.Normal);
						//reflectance *= maskingShadowing;
					}
					else
					{
						// Rough plastic
						if (rng.NextFloat() < Schlick(incidentCosine, IndexOfRefraction))
						{
							// Glossy reflection
							scattered = new Ray(rec.Point, reflect(ray.Direction, roughNormal), ray.Time);
						}
						else
						{
							// Lambertian diffuse
							float3 randomDirection = rng.OnCosineWeightedHemisphere(rec.Normal);
							scattered = new Ray(rec.Point, randomDirection, ray.Time);
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

		static float MicrofacetDistribution(float alpha, float cosine)
		{
			if (cosine <= 0)
				return 0;

			float sqAlpha = alpha * alpha;
			float sqCosine = cosine * cosine;
			float sqTanTheta = (1 - sqCosine) / sqCosine;
			float x = sqAlpha + sqTanTheta;
			return sqAlpha / (PI * sqCosine * sqCosine * x * x);
		}

		// Float TrowbridgeReitzDistribution::Lambda(const Vector3f &w) const {
		// 	Float absTanTheta = std::abs(TanTheta(w));
		// 	if (std::isinf(absTanTheta)) return 0.;
		// 		<<Compute alpha for direction w>>
		// 		Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
		// 	return (-1 + std::sqrt(1.f + alpha2Tan2Theta)) / 2;
		// }
	}
}