using System;
using JetBrains.Annotations;
using Unity.Burst;
using Unity.Mathematics;
using UnityEditor.PackageManager;
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

		float Roughness => Parameter; // for metals

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
			out float3 reflectance, out Ray scattered)
		{
			switch (Type)
			{
				case MaterialType.Lambertian:
				{
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					float3 randomDirection = rng.OnUniformHemisphere(rec.Normal);
					scattered = new Ray(rec.Point, randomDirection, ray.Time);
					return true;
				}

				case MaterialType.Metal:
				{
					// TODO: extract PDF so we can do explicit sampling
					// TODO: use the "part 2" equations instead
					// TODO: avoid the swap by having a y-up tangent space

					float3 outgoingDirection = MathExtensions.WorldToTangentSpace(-ray.Direction, rec.Normal);
					Util.Swap(ref outgoingDirection.y, ref outgoingDirection.z); // remap z to y

					if (ImportanceSampleGgxD(Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData),
						ref rng, outgoingDirection, Roughness, out float3 toLight, out reflectance))
					{
						Util.Swap(ref toLight.y, ref toLight.z); // remap y to z
						float3 scatterDirection = MathExtensions.TangentToWorldSpace(toLight, rec.Normal);
						scattered = new Ray(rec.Point, scatterDirection, ray.Time);
						return true;
					}

					scattered = default;
					return false;
				}

				case MaterialType.Dielectric:
				{
					float refractiveIndex = Parameter;
					float3 reflected = reflect(ray.Direction, rec.Normal);
					reflectance = 1;
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
					reflectance = Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData);
					return true;

				default:
					reflectance = default;
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

		static float Schlick(float radians, float refractiveIndex)
		{
			float r0 = (1 - refractiveIndex) / (1 + refractiveIndex);
			r0 *= r0;
			float exponential = pow(1 - radians, 5);
			return r0 + (1 - r0) * exponential;
		}

		// ------ GGX from https://schuttejoe.github.io/post/ggximportancesamplingpart1/

		static float3 Schlick(float radians, float3 r0)
		{
			float exponential = pow(1.0f - radians, 5);
			return r0 + (1 - r0) * exponential;
		}

		static float SmithGgxMaskingShadowing(float3 wi, float3 wo, float a2)
		{
			float nDotL = wi.y;
			float nDotV = wo.y;

			float denomA = nDotV * sqrt(a2 + (1.0f - a2) * nDotL * nDotL);
			float denomB = nDotL * sqrt(a2 + (1.0f - a2) * nDotV * nDotV);

			return 2.0f * nDotL * nDotV / (denomA + denomB);
		}

		static bool ImportanceSampleGgxD(float3 specularColor, ref Random rng, float3 wo,
			float roughness, out float3 wi, out float3 reflectance)
		{
			float a = roughness;
			float a2 = a * a;

			// generate uniform random variables between 0 and 1
			float2 e = rng.NextFloat2();

			// calculate theta and phi for our microfacet normal wm by importance sampling the Ggx distribution of normals
			float theta = acos(sqrt((1.0f - e.x) / ((a2 - 1.0f) * e.x + 1.0f)));
			float phi = 2 * PI * e.y;

			// convert from spherical to Cartesian coordinates
			float3 wm = MathExtensions.SphericalToCartesian(theta, phi);

			// calculate wi by reflecting wo about wm
			wi = 2.0f * dot(wo, wm) * wm - wo;

			// ensure our sample is in the upper hemisphere
			// since we are in tangent space with a y-up coordinate, dot(n, wi) simply maps to wi.y
			if (wi.y > 0.0f && dot(wi, wm) > 0.0f)
			{
				float dotWiWm = dot(wi, wm);

				// calculate the reflectance to multiply by the energy retrieved in direction wi
				float3 f = Schlick(dotWiWm, specularColor);
				float g = SmithGgxMaskingShadowing(wi, wo, a2);
				float weight = abs(dot(wo, wm))
				               / (wo.y * wm.y);

				reflectance = f * g * weight;
				return true;
			}

			reflectance = 0;
			return false;
		}

		// https://hal.archives-ouvertes.fr/hal-01509746/document
		static float3 GgxVndf(float3 wo, float roughness, float u1, float u2)
		{
			float3 v = normalize(float3(wo.x * roughness, wo.y, wo.z * roughness));

			// build an orthonormal basis with v, t1, and t2
			MathExtensions.GetOrthonormalBasis(v, out float3 t1, out float3 t2);

			// choose a point on a disk with each half of the disk weighted proportionally to its projection onto direction v
			float a = 1.0f / (1.0f + v.y);
			float r = sqrt(u1);
			float phi = (u2 < a) ? (u2 / a) * PI : PI + (u2 - a) / (1.0f - a) * PI;
			sincos(phi, out float sinPhi, out float cosPhi);
			float p1 = r * cosPhi;
			float p2 = r * sinPhi * (u2 < a ? 1.0f : v.y);

			// calculate the normal in this stretched tangent space
			float3 n = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

			// unstretch and normalize the normal
			return normalize(float3(roughness * n.x, max(0.0f, n.y), roughness * n.z));
		}

		// ------ GGX from https://schuttejoe.github.io/post/ggximportancesamplingpart2/

		static float SmithGgxMasking(float3 wi, float3 wo, float a2)
		{
			float nDotV = wo.y;
			float denomC = sqrt(a2 + (1.0f - a2) * nDotV * nDotV) + nDotV;

			return 2.0f * nDotV / denomC;
		}

		static void ImportanceSampleGgxVdn(float3 specularColor, float roughness, ref Random rng,
			float3 wo, out float3 wi, out float3 reflectance)
		{
			float a = roughness;
			float a2 = a * a;

			float2 r = rng.NextFloat2();
			float3 wm = GgxVndf(wo, a, r.x, r.y);

			wi = reflect(wm, wo);

			if (wi.y > 0.0f)
			{
				float3 f = Schlick(dot(wi, wm), specularColor);
				float g1 = SmithGgxMasking(wi, wo, a2);
				float g2 = SmithGgxMaskingShadowing(wi, wo, a2);

				reflectance = f * (g2 / g1);
			}
			else
				reflectance = 0;
		}
	}
}