using System;
using JetBrains.Annotations;
using Unity.Mathematics;
using Util;
using static Unity.Mathematics.math;

namespace Runtime
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

	readonly struct Material
	{
		public readonly MaterialType Type;
		public readonly Texture Texture, Roughness;
		public readonly float2 TextureScale;
		public readonly float Parameter;

		float RefractiveIndex => Parameter; // for Dielectric
		public float Density => Parameter; // for ProbabilisticVolume

		public Material(MaterialType type, float2 textureScale, Texture albedo = default,
			Texture emission = default, Texture roughness = default, float refractiveIndex = 1, float density = 1) : this()
		{
			Type = type;
			TextureScale = textureScale;

			switch (type)
			{
				case MaterialType.Lambertian: Texture = albedo; break;
				case MaterialType.Metal: Roughness = roughness; Texture = albedo; break;
				case MaterialType.Dielectric: Roughness = roughness; Parameter = refractiveIndex; Texture = albedo; break;
				case MaterialType.DiffuseLight: Texture = emission; break;
				case MaterialType.ProbabilisticVolume: Parameter = density; Texture = albedo; break;
			}
		}

		[Pure]
		public bool ProbabilisticHit(ref float hitDistance, ref RandomSource rng)
		{
			if (Type != MaterialType.ProbabilisticVolume)
				return false;

			float volumeHitDistance = -(1 / Density) * log(rng.NextFloat());

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
			switch (Type)
			{
				case MaterialType.Lambertian:
				{
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinNoise);
					//float3 randomDirection = rng.OnCosineWeightedHemisphere(rec.Normal);
					float3 randomDirection = rng.OnUniformHemisphere(rec.Normal);
					scattered = new Ray(rec.Point, randomDirection, ray.Time);
					break;
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
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinNoise);
					float roughness = Roughness.Value(rec.Point, rec.Normal, TextureScale, perlinNoise).x;
					scattered = new Ray(rec.Point, normalize(reflected + roughness * rng.NextFloat3Direction()), ray.Time);
					break;
				}

				case MaterialType.Dielectric:
				{
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinNoise);

					float roughness = Roughness.Value(rec.Point, rec.Normal, TextureScale, perlinNoise).x;
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
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinNoise);
					break;

				default:
					reflectance = default;
					scattered = default;
					break;
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
					// Disabled because it currently does not work right
					//return GgxMicrofacet.Pdf(incomingLightDirection, outgoingLightDirection, geometricNormal, Roughness);

				case MaterialType.DiffuseLight:
					return 0; // Doesn't matter, because there is no scattering or albedo

				default: throw new NotImplementedException();
			}
		}

		[Pure]
		public float3 Emit(float3 position, float3 normal, PerlinNoise perlinNoise)
		{
			switch (Type)
			{
				case MaterialType.DiffuseLight: return Texture.Value(position, normal, TextureScale, perlinNoise);
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
					case MaterialType.Metal: return Roughness.Type == TextureType.Constant && all(Roughness.MainColor.AlmostEquals(0));
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