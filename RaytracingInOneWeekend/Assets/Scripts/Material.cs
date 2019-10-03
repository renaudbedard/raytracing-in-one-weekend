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
					float3 randomDirection = rng.OnUniformHemisphere(rec.Normal);
					scattered = new Ray(rec.Point, randomDirection, ray.Time);
					return true;
				}

				case MaterialType.Metal:
				{
					// TODO: extract PDF so we can do explicit sampling

					float3 outgoingDirection = MathExtensions.WorldToTangentSpace(-ray.Direction, rec.Normal);

					if (ImportanceSampleGgxD(Texture.Value(rec.Point, rec.Normal, TextureScale, perlinData),
						Roughness, ref rng, outgoingDirection, out float3 toLight, out reflectance))
					{
						float3 scatterDirection = MathExtensions.TangentToWorldSpace(toLight, rec.Normal);
						scattered = new Ray(rec.Point, scatterDirection, ray.Time);
						return true;
					}

					scattered = default;
					return false;
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
		public float ScatteringPdf(float3 outgoingDirection, float3 scatteredDirection, float3 geometricNormal)
		{
			switch (Type)
			{
				case MaterialType.Lambertian:
					return max(dot(geometricNormal, scatteredDirection) / PI, 0);

				case MaterialType.ProbabilisticVolume:
					return 1.0f / (4.0f * PI);

				case MaterialType.Metal:
					outgoingDirection = MathExtensions.WorldToTangentSpace(outgoingDirection, geometricNormal);
					scatteredDirection = MathExtensions.WorldToTangentSpace(scatteredDirection, geometricNormal);
					float3 halfVector = normalize(outgoingDirection + scatteredDirection);
					float eoh = abs(dot(outgoingDirection, halfVector)); // TODO: no idea what EoH stands for
					return GgxIsotropicNdf(Roughness, halfVector) * abs(halfVector.y) / (4.0f * eoh);

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

		// ------ GGX from https://schuttejoe.github.io/post/ggximportancesamplingpart1/

		static float3 Schlick(float radians, float3 r0)
		{
			float exponential = pow(1 - radians, 5);
			return r0 + (1 - r0) * exponential;
		}

		static float SmithGgxMaskingShadowing(float3 wi, float3 wo, float a2)
		{
			float nDotL = wi.y;
			float nDotV = wo.y;

			float denomA = nDotV * sqrt(a2 + (1 - a2) * nDotL * nDotL);
			float denomB = nDotL * sqrt(a2 + (1 - a2) * nDotV * nDotV);

			return 2 * nDotL * nDotV / (denomA + denomB);
		}

		// Probability of facet with specific normal (h)
		static float GgxIsotropicNdf(float roughness, float3 h)
		{
			// from https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html (figure 13)
			float sqRoughness = roughness * roughness;
			float sqCosTheta = h.y * h.y;
			float sqrtDenomOverPi = sqCosTheta * (sqRoughness - 1) + 1;
			return sqRoughness / (PI * sqrtDenomOverPi * sqrtDenomOverPi);
		}

		static bool ImportanceSampleGgxD(float3 specularColor, float roughness, ref Random rng, float3 wo,
			out float3 wi, out float3 reflectance)
		{
			float a = roughness;
			float a2 = a * a;

			// generate uniform random variables between 0 and 1
			float2 e = rng.NextFloat2();

			// calculate theta and phi for our microfacet normal wm by importance sampling the Ggx distribution of normals
			float theta = acos(sqrt((1 - e.x) / ((a2 - 1) * e.x + 1)));
			float phi = 2 * PI * e.y;

			// alternate from : // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
			// theta = atan( alphaU * sqrt( bs.v / ( 1.0f - bs.v )) );

			// convert from spherical to Cartesian coordinates
			float3 wm = MathExtensions.SphericalToCartesian(theta, phi);

			// calculate wi by reflecting wo about wm
			wi = 2 * dot(wo, wm) * wm - wo;

			// ensure our sample is in the upper hemisphere
			// since we are in tangent space with a z-up coordinate, dot(n, wi) simply maps to wi.z
			if (wi.y > 0 && dot(wi, wm) > 0)
			{
				float dotWiWm = dot(wi, wm);

				// calculate the reflectance to multiply by the energy retrieved in direction wi
				float3 f = Schlick(dotWiWm, specularColor);
				float g = SmithGgxMaskingShadowing(wi, wo, a2);
				float weight = abs(dot(wo, wm)) / (wo.y * wm.y);

				reflectance = f * g * weight;
				return true;
			}

			reflectance = 0;
			return false;
		}
	}
}