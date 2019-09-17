using System;
using JetBrains.Annotations;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine.Assertions;
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
		DiffuseLight,
		Isotropic
	}

	struct Material
	{
		public readonly MaterialType Type;
		public readonly Texture Texture;
		public readonly float2 TextureScale;
		public readonly float Parameter;

		public Material(MaterialType type, float2 textureScale, Texture albedo = default,
			Texture emission = default, float fuzz = 0, float refractiveIndex = 1, float density = 1) : this()
		{
			Type = type;
			TextureScale = textureScale;

			switch (type)
			{
				case MaterialType.Lambertian: Texture = albedo; break;
				case MaterialType.Metal: Parameter = saturate(fuzz); Texture = albedo; break;
				case MaterialType.Dielectric: Parameter = refractiveIndex; break;
				case MaterialType.DiffuseLight: Texture = emission; break;
				case MaterialType.Isotropic: Parameter = density; Texture = albedo; break;
			}
		}

		[Pure]
		public bool Scatter(Ray r, HitRecord rec, ref Random rng, PerlinData perlinData,
			out float3 albedo, out Ray scattered)
		{
			switch (Type)
			{
				case MaterialType.Lambertian:
				{
					albedo = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					float3 randomDirection = rng.OnUniformHemisphere(rec.Normal);
					scattered = new Ray(rec.Point, randomDirection, r.Time);
					return true;
				}

				case MaterialType.Metal:
				{
					float fuzz = Parameter;
					float3 reflected = reflect(r.Direction, rec.Normal);
					albedo = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					scattered = new Ray(rec.Point, normalize(reflected + fuzz * rng.NextFloat3Direction()), r.Time);
					return true;
				}

				case MaterialType.Dielectric:
				{
					float refractiveIndex = Parameter;
					float3 reflected = reflect(r.Direction, rec.Normal);
					albedo = 1;
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

				case MaterialType.Isotropic:
					scattered = new Ray(rec.Point, rng.NextFloat3Direction());
					albedo = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					return true;

				default:
					albedo = default;
					scattered = default;
					return false;
			}
		}

		[BurstDiscard]
		void Validate(Ray r, HitRecord rec)
		{
			Assert.IsTrue(length(r.Direction).AlmostEquals(1),
				$"Scatter ray direction was assumed to be unit-length; length was {length(r.Direction):0.#######}");
			Assert.IsTrue(length(rec.Normal).AlmostEquals(1),
				$"HitRecord normal was assumed to be unit-length; length was {length(rec.Normal):0.#######}");
		}

		[Pure]
		public float ScatteringPdf(HitRecord rec, Ray scattered)
		{
			Validate(scattered, rec);
			switch (Type)
			{
				case MaterialType.Lambertian: return max(dot(rec.Normal, scattered.Direction) / PI, 0);
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
					case MaterialType.Dielectric:
					case MaterialType.Metal:
						return Parameter.AlmostEquals(0);
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
			return r0 + (1 - r0) * pow((1 - cosine), 5);
		}
	}
}