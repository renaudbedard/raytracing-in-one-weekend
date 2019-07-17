using Unity.Burst;
using Unity.Collections;
using Unity.Collections.Experimental;
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
		[ReadOnly] public float3 SkyBottomColor;
		[ReadOnly] public float3 SkyTopColor;
		[ReadOnly] public uint Seed;
#if MANUAL_AOSOA
		[ReadOnly] public AosoaSpheres World;
#elif MANUAL_SOA
		[ReadOnly] public SoaSpheres World;
#elif UNITY_SOA
		[ReadOnly] public NativeArrayFullSOA<Sphere> World;
#else
		[ReadOnly] public NativeArray<Primitive> World;
#endif
#if BUFFERED_MATERIALS || UNITY_SOA
		[ReadOnly] public NativeArray<Material> Material;
#endif
		[ReadOnly] public NativeArray<float4> InputSamples;

		[WriteOnly] public NativeArray<float4> OutputSamples;
		[WriteOnly] public NativeArray<uint> OutputRayCount;

		bool Color(Ray r, uint depth, Random rng, out float3 color, ref uint rayCount)
		{
			rayCount++;

			if (World.Hit(r, 0.001f, float.PositiveInfinity, out HitRecord rec))
			{
#if BUFFERED_MATERIALS || UNITY_SOA
				if (depth < TraceDepth && Material[rec.MaterialIndex].Scatter(r, rec, rng, out float3 attenuation, out Ray scattered))
#else
				if (depth < TraceDepth && rec.Material.Scatter(r, rec, rng, out float3 attenuation, out Ray scattered))
#endif
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
			color = lerp(SkyBottomColor, SkyTopColor, t);
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
				finalColor = Input[index].xyz / realSampleCount;

			Output[index] = half4(half3(finalColor), half(1));
		}
	}
}