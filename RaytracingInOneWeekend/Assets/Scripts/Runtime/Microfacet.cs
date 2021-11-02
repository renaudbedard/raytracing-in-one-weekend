using Unity.Mathematics;
using Util;
using static Unity.Mathematics.math;

namespace Runtime
{
	public static class Microfacet
	{
		public static float TorranceSparrowBrdf(float3 wi, float3 wo, float3 normal, float roughness, float fresnel)
		{
			float3 wh = normalize(wi + wo);
			float absCosThetaO = abs(dot(wo, normal));
			float absCosThetaI = abs(dot(wi, normal));

			return TrowbridgeReitz.D(wh, normal, roughness) * TrowbridgeReitz.G(wi, wo, normal, roughness) *
				fresnel / (4 * absCosThetaI * absCosThetaO);
		}

		public static class TrowbridgeReitz
		{
			public static float D(float3 wh, float3 normal, float roughness)
			{
				Tools.GetOrthonormalBasis(normal, out float3 tangent, out float3 bitangent);
				float alpha = RoughnessToAlpha(roughness);
				float sqAlpha = alpha * alpha;

				float cosTheta = dot(normal, wh);
				float sqCosTheta = cosTheta * cosTheta;
				float sqSinTheta = max(0, 1 - sqCosTheta);
				float sinTheta = sqrt(sqSinTheta);
				float tanTheta = sinTheta / cosTheta;
				float sqTanTheta = tanTheta * tanTheta;
				float cosPhi = sinTheta == 0 ? 1 : clamp(dot(wh, tangent) / sinTheta, -1, 1);
				float sinPhi = sinTheta == 0 ? 1 : clamp(dot(wh, bitangent) / sinTheta, -1, 1);

				if (isinf(sqTanTheta))
					return 0;

				float e = (cosPhi * cosPhi / sqAlpha + sinPhi * sinPhi / sqAlpha) * sqTanTheta;
				return 1 / (PI * sqAlpha * sqCosTheta * sqCosTheta * (1 + e) * (1 + e));
			}

			public static float G(float3 wi, float3 wo, float3 normal, float roughness)
			{
				return 1 / (1 + Lambda(wo, normal, roughness) + Lambda(wi, normal, roughness));
			}

			public static float Lambda(float3 w, float3 normal, float roughness)
			{
				float cosTheta = dot(normal, w);
				float sqCosTheta = cosTheta * cosTheta;
				float sqSinTheta = max(0, 1 - sqCosTheta);
				float sinTheta = sqrt(sqSinTheta);
				float tanTheta = sinTheta / cosTheta;

				float absTanTheta = abs(tanTheta);
				if (isinf(absTanTheta))
					return 0;

				float alpha = RoughnessToAlpha(roughness);

				float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
				return (-1 + sqrt(1 + alpha2Tan2Theta)) / 2;
			}

			static float RoughnessToAlpha(float roughness)
			{
				roughness = max(roughness, 1e-3f);
				float x = log(roughness);
				return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
				       0.000640711f * x * x * x * x;
			}
		}
	}
}