using Unity.Mathematics;
using Util;
using static Unity.Mathematics.math;

namespace Runtime
{
	public static class Microfacet
	{
		public static float SmithMaskingShadowing(float3 w, float roughness, float3 normal)
		{
			return 1 / (1 + Beckmann.Lambda(w, roughness, normal));
		}

		public static float TorranceSparrowBrdf()
		{
			// TODO
			return default;
		}

		public static class Beckmann
		{
			public static float D(float3 w, float roughness, float3 normal)
			{
				Tools.GetOrthonormalBasis(normal, out float3 tangent, out float3 bitangent);
				float alpha = RoughnessToAlpha(roughness);

				float cosTheta = dot(normal, w);
				float sqCosTheta = cosTheta * cosTheta;
				float sqSinTheta = max(0, 1 - sqCosTheta);
				float sinTheta = sqrt(sqSinTheta);
				float tanTheta = sinTheta / cosTheta;
				float sqTanTheta = tanTheta * tanTheta;
				float cosPhi = sinTheta == 0 ? 1 : clamp(dot(w, tangent) / sinTheta, -1, 1);
				float sinPhi = sinTheta == 0 ? 1 : clamp(dot(w, bitangent) / sinTheta, -1, 1);

				if (isinf(sqTanTheta))
					return 0;

				return exp(-sqTanTheta * (cosPhi * cosPhi / (alpha * alpha) + sinPhi * sinPhi / (alpha * alpha))) /
				       (PI * alpha * alpha * sqCosTheta * sqCosTheta);
			}

			public static float Lambda(float3 w, float roughness, float3 normal)
			{
				float cosine = dot(normal, w);
				float sqCosine = cosine * cosine;
				float sqSine = max(0, 1 - sqCosine);
				float sine = sqrt(sqSine);
				float tangent = sine / cosine;

				float absTanTheta = abs(tangent);
				if (isinf(absTanTheta))
					return 0;

				float alpha = RoughnessToAlpha(roughness);

				float a = 1 / (alpha * absTanTheta);
				if (a >= 1.6f)
					return 0;

				return (1 - 1.259f * a + 0.396f * a * a) /
				       (3.535f * a + 2.181f * a * a);
			}

			static float RoughnessToAlpha(float roughness)
			{
				roughness = max(roughness, 1e-3f);
				float x = log(roughness);
				return 1.62142f + 0.819955f * x + 0.1734f * x * x +
				       0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
			}
		}

		public static class TrowbridgeReitz
		{
			public static float Lambda(float3 w, float roughness, float3 normal)
			{
				float cosine = dot(normal, w);
				float sqCosine = cosine * cosine;
				float sqSine = max(0, 1 - sqCosine);
				float sine = sqrt(sqSine);
				float tangent = sine / cosine;

				float absTanTheta = abs(tangent);
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