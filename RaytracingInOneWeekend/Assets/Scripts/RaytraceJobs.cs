using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct AccumulateJob : IJobParallelFor
	{
		[ReadOnly] public uint2 Size;
		[ReadOnly] public Camera Camera;
		[ReadOnly] public uint SampleCount;
		[ReadOnly] public uint TraceDepth;
		[ReadOnly] public uint Seed;
#if AOSOA_SPHERES
		[ReadOnly] public AosoaSpheres World;
#elif SOA_SPHERES
		[ReadOnly] public SoaSpheres World;
#else
		[ReadOnly] public NativeArray<Primitive> World;
#endif
#if BUFFERED_MATERIALS
		[ReadOnly] public NativeArray<Material> Materials;
#endif
		[ReadOnly] public NativeArray<float4> InputSamples;

		[WriteOnly] public NativeArray<float4> OutputSamples;
		[WriteOnly] public NativeArray<uint> OutputRayCount;

		bool Color(Ray r, uint depth, Random rng, out float3 color, ref uint rayCount)
		{
			rayCount++;

			if (World.Hit(r, 0.001f, float.PositiveInfinity, out HitRecord rec))
			{
				if (depth < TraceDepth &&
#if BUFFERED_MATERIALS 
				    Materials[rec.MaterialIndex]
#else
					rec.Material
#endif
					.Scatter(r, rec, rng, out float3 attenuation, out Ray scattered))
				{
					if (Color(scattered, depth + 1, rng, out float3 scatteredColor, ref rayCount))
					{
						color = attenuation * scatteredColor;
						return true;
					}
				}
				color = default;
				return false;
			}

			float3 unitDirection = normalize(r.Direction);
			float t = 0.5f * (unitDirection.y + 1);
			color = lerp(1, float3(0.5f, 0.7f, 1), t);
			return true;
		}

		public void Execute(int index)
		{
			uint2 coordinates = uint2(
				(uint) (index % Size.x), // column
				(uint) (index / Size.x)  // row
			);

			float4 lastValue = InputSamples[index];

			float3 colorAcc = lastValue.xyz;
			int sampleCount = (int) lastValue.w;

			var rng = new Random(Seed + (uint) index * 0x7383ED49u);
			uint rayCount = 0;

			for (int s = 0; s < SampleCount; s++)
			{
				float2 normalizedCoordinates = (coordinates + rng.NextFloat2()) / Size; // (u, v)
				Ray r = Camera.GetRay(normalizedCoordinates, rng);
				if (Color(r, 0, rng, out float3 sampleColor, ref rayCount))
				{
					colorAcc += sampleColor;
					sampleCount++;
				}
			}

			OutputSamples[index] = float4(colorAcc, sampleCount);
			OutputRayCount[index] = rayCount;
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct CombineJob : IJobParallelFor
	{
		static readonly float3 NoSamplesColor = new float3(1, 0, 1);

		[ReadOnly] public NativeArray<float4> Input;
		[WriteOnly] public NativeArray<half4> Output;

		public void Execute(int index)
		{
			var realSampleCount = (int) Input[index].w;

			float3 finalColor;
			if (realSampleCount == 0)
				finalColor = NoSamplesColor;
			else
			{
				finalColor = Input[index].xyz / realSampleCount;
				finalColor = sqrt(finalColor);
			}

			Output[index] = half4(half3(finalColor), half(1));
		}
	}
}