using Unity.Mathematics;
using Util;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace Runtime
{
	enum NoiseColor
	{
		White,
		Blue,
		SpatioTemporalBlue
	}

	struct RandomSource
	{
		readonly NoiseColor noiseColor;

		Random whiteNoise;
		PerPixelBlueNoise blueNoise;
		float randomEvents;

		public RandomSource(NoiseColor noiseColor, Random whiteNoise, PerPixelBlueNoise blueNoise)
		{
			this.noiseColor = noiseColor;
			this.whiteNoise = whiteNoise;
			this.blueNoise = blueNoise;
			randomEvents = 0;
		}

		public float RandomEvents
		{
			get => randomEvents;
			set => randomEvents = value;
		}

		// from : https://programming.guide/random-point-within-circle.html
		public float2 InUnitDisk()
		{
			float theta = default, radius = default;
			switch (noiseColor)
			{
				case NoiseColor.White:
					theta = whiteNoise.NextFloat(0, 2 * PI);
					radius = sqrt(whiteNoise.NextFloat());
					break;

				case NoiseColor.Blue:
					theta = blueNoise.NextFloat(0, 2 * PI);
					radius = sqrt(blueNoise.NextFloat());
					break;
			}

			sincos(theta, out float sinTheta, out float cosTheta);
			return radius * float2(cosTheta, sinTheta);
		}

		public float3 OnCosineWeightedHemisphere(float3 normal)
		{
			float2 uv = default;
			switch (noiseColor)
			{
				case NoiseColor.White: uv = whiteNoise.NextFloat2(); break;
				case NoiseColor.Blue: uv = blueNoise.NextFloat2();	break;
			}

			// uniform sampling of a cosine-weighted hemisphere
			// from : https://cg.informatik.uni-freiburg.de/course_notes/graphics2_08_renderingEquation.pdf (inversion method, page 47)
			// same algorithm used here : http://www.rorydriscoll.com/2009/01/07/better-sampling/
			float u = uv.x;
			float radius = sqrt(u);
			float theta = uv.y * 2 * PI;
			sincos(theta, out float sinTheta, out float cosTheta);
			float2 xz = radius * float2(cosTheta, sinTheta);
			float3 tangentSpaceDirection = float3(xz.x, sqrt(1 - u), xz.y);

			return Tools.TangentToWorldSpace(tangentSpaceDirection, normal);
		}

		public float3 OnUniformHemisphere(float3 normal)
		{
			float2 uv = default;
			switch (noiseColor)
			{
				case NoiseColor.White: uv = whiteNoise.NextFloat2(); break;
				case NoiseColor.Blue: uv = blueNoise.NextFloat2();	break;
			}

			// uniform sampling of a hemisphere
			// from : https://cg.informatik.uni-freiburg.de/course_notes/graphics2_08_renderingEquation.pdf (inversion method, page 42)
			float u = uv.x;
			float radius = sqrt(2 * u - u * u);
			float theta = uv.y * 2 * PI;
			sincos(theta, out float sinTheta, out float cosTheta);
			float2 xz = radius * float2(cosTheta, sinTheta);
			float3 tangentSpaceDirection = float3(xz.x, 1 - u, xz.y);

			return Tools.TangentToWorldSpace(tangentSpaceDirection, normal);
		}

		public float3 NextFloat3Direction()
		{
			float2 rnd = default;
			switch (noiseColor)
			{
				case NoiseColor.White: rnd = whiteNoise.NextFloat2(); break;
				case NoiseColor.Blue: rnd = blueNoise.NextFloat2();	break;
			}

			float z = rnd.x * 2.0f - 1.0f;
			float r = sqrt(max(1.0f - z * z, 0.0f));
			float angle = rnd.y * PI * 2.0f;
			sincos(angle, out float s, out float c);
			return float3(c * r, s * r, z);
		}

		public float NextFloat()
		{
			switch (noiseColor)
			{
				case NoiseColor.White: return whiteNoise.NextFloat();
				case NoiseColor.Blue: return blueNoise.NextFloat();
			}
			return default;
		}

		public float2 NextFloat2()
		{
			switch (noiseColor)
			{
				case NoiseColor.White: return whiteNoise.NextFloat2();
				case NoiseColor.Blue: return blueNoise.NextFloat2();
			}
			return default;
		}

		public float2 NextFloat2(float2 from, float2 to)
		{
			switch (noiseColor)
			{
				case NoiseColor.White: return whiteNoise.NextFloat2(from, to);
				case NoiseColor.Blue: return blueNoise.NextFloat2(from, to);
			}
			return default;
		}

		public int NextInt(int from, int to)
		{
			switch (noiseColor)
			{
				case NoiseColor.White: return whiteNoise.NextInt(from, to);
				case NoiseColor.Blue: return blueNoise.NextInt(from, to);
			}
			return default;
		}
	}
}