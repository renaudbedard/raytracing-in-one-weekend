using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

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

#if BASIC
		[ReadOnly] public NativeArray<Entity> World;
#elif SOA_SIMD
		[ReadOnly] public SoaSpheres World;
#elif AOSOA_SIMD
		[ReadOnly] public AosoaSpheres World;
#elif BVH
		[ReadOnly] public BvhNode World;
#endif

#if BUFFERED_MATERIALS
		[ReadOnly] public NativeArray<Material> Material;
#endif

		[ReadOnly] public NativeArray<float4> InputSamples;
		[WriteOnly] public NativeArray<float4> OutputSamples;
		[WriteOnly] public NativeArray<uint> OutputRayCount;

#if BVH_ITERATIVE
#pragma warning disable 649
		[NativeSetThreadIndex] int threadIndex;
#pragma warning restore 649

		public int ThreadCount;
		public NativeArray<IntPtr> NodeWorkingBuffer;
		public NativeArray<Entity> EntityWorkingBuffer;
#if BVH_SIMD
		public NativeArray<float4> VectorWorkingBuffer;
#endif

		public unsafe struct WorkingArea
		{
			public BvhNode** Nodes;
			public Entity* Entities;
#if BVH_SIMD
			public float4* Vectors;
#endif
		}
#endif

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

#if BVH_ITERATIVE
			// for some reason, thread indices are [1, ProcessorCount] instead of [0, ProcessorCount[
			int actualThreadIndex = threadIndex - 1;
			WorkingArea workingArea;
			unsafe
			{
				workingArea = new WorkingArea
				{
					Nodes = (BvhNode**) NodeWorkingBuffer.GetUnsafeReadOnlyPtr() +
					        actualThreadIndex * (NodeWorkingBuffer.Length / ThreadCount),
					Entities = (Entity*) EntityWorkingBuffer.GetUnsafeReadOnlyPtr() +
					           actualThreadIndex * (EntityWorkingBuffer.Length / ThreadCount),
#if BVH_SIMD
					Vectors = (float4*) VectorWorkingBuffer.GetUnsafeReadOnlyPtr() +
					          actualThreadIndex * (VectorWorkingBuffer.Length / ThreadCount)
#endif
				};
			}
#endif

			for (int s = 0; s < SampleCount; s++)
			{
				float2 normalizedCoordinates = (coordinates + rng.NextFloat2()) / Size; // (u, v)
				Ray r = Camera.GetRay(normalizedCoordinates, rng);

#if BVH_ITERATIVE
				if (Color(r, 0, rng, workingArea, out float3 sampleColor, ref rayCount))
#else
				if (Color(r, 0, rng, out float3 sampleColor, ref rayCount))
#endif
				{
					colorAcc += sampleColor;
					sampleCount++;
				}
			}

			OutputSamples[index] = float4(colorAcc, sampleCount);
			OutputRayCount[index] = rayCount;
		}

#if BVH_ITERATIVE
		bool Color(Ray r, uint depth, Random rng, WorkingArea wa, out float3 color, ref uint rayCount)
#else
		bool Color(Ray r, uint depth, Random rng, out float3 color, ref uint rayCount)
#endif
		{
			rayCount++;

#if BVH_ITERATIVE
			if (World.Hit(r, 0.001f, float.PositiveInfinity, wa, out HitRecord rec))
#else
			if (World.Hit(r, 0.001f, float.PositiveInfinity, out HitRecord rec))
#endif
			{
#if BUFFERED_MATERIALS
				if (depth < TraceDepth &&
				    Material[rec.MaterialIndex].Scatter(r, rec, rng, out float3 attenuation, out Ray scattered))
#else
				if (depth < TraceDepth && rec.Material.Scatter(r, rec, rng, out float3 attenuation, out Ray scattered))
#endif
				{
#if BVH_ITERATIVE
					if (Color(scattered, depth + 1, rng, wa, out float3 scatteredColor, ref rayCount))
#else
					if (Color(scattered, depth + 1, rng, out float3 scatteredColor, ref rayCount))
#endif
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