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
				case MaterialType.Isotropic: Parameter = density; Texture = albedo; break;
			}
		}

		[Pure]
		public bool Scatter(Ray ray, HitRecord rec, ref Random rng, PerlinData perlinData,
			out float3 albedo, out Ray scattered)
		{
			switch (Type)
			{
				case MaterialType.Lambertian:
				{
					albedo = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					float3 randomDirection = rng.OnUniformHemisphere(rec.Normal);
					scattered = new Ray(rec.Point, randomDirection, ray.Time);
					return true;
				}

				case MaterialType.Metal:
				{
					albedo = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);

					// https://computergraphics.stackexchange.com/questions/7656/importance-sampling-microfacet-ggx
					// r0 := rnd.Float64()
					// r1 := rnd.Float64()
					// a := m.Roughness * m.Roughness
					// a2 := a * a
					// theta := math.Acos(math.Sqrt((1 - r0) / ((a2-1)*r0 + 1)))
					// phi := 2 * math.Pi * r1
					// x := math.Sin(theta) * math.Cos(phi)
					// y := math.Cos(theta)
					// z := math.Sin(theta) * math.Sin(phi)
					// wm := geom.Vector3{x, y, z}.Unit()
					// wi := wo.Reflect2(wm)
					// return wi

					float2 r = rng.NextFloat2();
					float a = Parameter * Parameter;
					float a2 = a * a;
					float theta = acos(sqrt((1 - r[0]) / ((a2 - 1) * r[0] + 1)));
					float phi = 2 * PI * r[1];
					sincos(float2(theta, phi), out float2 sinThetaPhi, out float2 cosThetaPhi);
					float3 wm = normalize(float3(sinThetaPhi.x * cosThetaPhi.y, cosThetaPhi.x, sinThetaPhi.x * sinThetaPhi.y));
					float3 wi = reflect(ray.Direction, wm);
					scattered = new Ray(rec.Point, wi);
					return true;
				}

				case MaterialType.Dielectric:
				{
					float refractiveIndex = Parameter;
					float3 reflected = reflect(ray.Direction, rec.Normal);
					albedo = 1;
					float niOverNt;
					float3 outwardNormal;
					float cosine;

					if (dot(ray.Direction, rec.Normal) > 0)
					{
						outwardNormal = -rec.Normal;
						niOverNt = refractiveIndex;
						cosine = refractiveIndex * dot(ray.Direction, rec.Normal);
					}
					else
					{
						outwardNormal = rec.Normal;
						niOverNt = 1 / refractiveIndex;
						cosine = -dot(ray.Direction, rec.Normal);
					}

					if (Refract(ray.Direction, outwardNormal, niOverNt, out float3 refracted))
					{
						float reflectProb = Schlick(cosine, refractiveIndex);
						scattered = new Ray(rec.Point, rng.NextFloat() < reflectProb ? reflected : refracted, ray.Time);
					}
					else
						scattered = new Ray(rec.Point, reflected, ray.Time);

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
				case MaterialType.Lambertian:
					return max(dot(rec.Normal, scattered.Direction) / PI, 0);

				case MaterialType.Isotropic:
					return 1.0f / (4.0f * PI);

				// https://computergraphics.stackexchange.com/questions/7656/importance-sampling-microfacet-ggx
				// case MaterialType.Metal:
				// 	wg := geom.Up
				// 	wm := wo.Half(wi)
				// 	a := m.Roughness * m.Roughness
				// 	a2 := a * a
				// 	cosTheta := wg.Dot(wm)
				// 	exp := (a2-1)*cosTheta*cosTheta + 1
				// 	D := a2 / (math.Pi * exp * exp)
				// 	return (D * wm.Dot(wg)) / (4 * wo.Dot(wm))

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
					case MaterialType.Metal: return Parameter.AlmostEquals(0);
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