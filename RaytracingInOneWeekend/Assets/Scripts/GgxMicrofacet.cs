using Unity.Mathematics;
using static Unity.Mathematics.math;
using static RaytracerInOneWeekend.MathExtensions;

namespace RaytracerInOneWeekend
{
	public static class GgxMicrofacet
	{
		public static float3 Schlick(float radians, float3 r0)
		{
			float exponential = pow(1 - radians, 5);
			return r0 + (1 - r0) * exponential;
		}

		public static float G1(float3 wi, float3 wo, float a2)
		{
			float nDotL = wi.y;
			float nDotV = wo.y;

			float denomA = nDotV * sqrt(a2 + (1 - a2) * nDotL * nDotL);
			float denomB = nDotL * sqrt(a2 + (1 - a2) * nDotV * nDotV);

			return 2 * nDotL * nDotV / (denomA + denomB);
		}

		// Probability of facet with specific normal (h)
		public static float Ndf(float roughness, float3 h)
		{
			// from https://www.tobias-franke.eu/log/2014/03/30/notes_on_importance_sampling.html (figure 13)
			float sqRoughness = roughness * roughness;
			float sqCosTheta = h.y * h.y;
			float sqrtDenomOverPi = sqCosTheta * (sqRoughness - 1) + 1;
			return sqRoughness / (PI * sqrtDenomOverPi * sqrtDenomOverPi);
		}

		public static bool ImportanceSample(float3 specularColor, float roughness, ref Random rng, float3 wo,
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
			float3 wm = SphericalToCartesian(theta, phi);

			// calculate wi by reflecting wo about wm
			wi = 2 * dot(wo, wm) * wm - wo;

			// ensure our sample is in the upper hemisphere
			// since we are in tangent space with a z-up coordinate, dot(n, wi) simply maps to wi.z
			if (wi.y > 0 && dot(wi, wm) > 0)
			{
				float dotWiWm = dot(wi, wm);

				// calculate the reflectance to multiply by the energy retrieved in direction wi
				float3 f = Schlick(dotWiWm, specularColor);
				float g = G1(wi, wo, a2);
				float weight = abs(dot(wo, wm)) / (wo.y * wm.y);

				reflectance = f * g * weight;
				return true;
			}

			reflectance = 0;
			return false;
		}

		public static float Pdf(float3 incomingLightDirection, float3 outgoingLightDirection, float3 geometricNormal, float roughness)
		{
			float3 wi = incomingLightDirection;
			float3 wo = outgoingLightDirection;

			wi = WorldToTangentSpace(wi, geometricNormal);
			wo = WorldToTangentSpace(wo, geometricNormal);

			float3 halfVector = normalize(wo + wi);
			float EoH = abs(dot(wo, halfVector)); // TODO: no idea what EoH stands for
			return Ndf(roughness, halfVector) * abs(halfVector.y) / (4.0f * EoH);
		}
	}
}