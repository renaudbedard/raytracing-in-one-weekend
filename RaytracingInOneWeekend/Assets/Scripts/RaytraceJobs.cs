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
		public struct Diagnostics
		{
			public int RayCount, BoundsHitCount, CandidateCount;
#pragma warning disable 649
			public int Unused;
#pragma warning restore 649
		}

#if BVH_ITERATIVE
		public unsafe struct WorkingArea
		{
			public BvhNode** Nodes;
			public Entity* Entities;
#if BVH_SIMD
			public float4* Vectors;
#endif
		}
#endif

		[ReadOnly] public float2 Size;
		[ReadOnly] public uint Seed;
		[ReadOnly] public int ThreadCount;
#if BVH_ITERATIVE
#pragma warning disable 649
		[NativeSetThreadIndex] [ReadOnly] int threadIndex;
#pragma warning restore 649
#endif
		[ReadOnly] public Camera Camera;
		[ReadOnly] public uint SampleCount;
		[ReadOnly] public uint TraceDepth;
		[ReadOnly] public float3 SkyBottomColor;
		[ReadOnly] public float3 SkyTopColor;

		[ReadOnly] public NativeArray<float4> InputSamples;

#if BASIC
		[ReadOnly] public NativeArray<Entity> World;
#elif SOA_SIMD
		[ReadOnly] public SoaSpheres World;
#elif AOSOA_SIMD
		[ReadOnly] public AosoaSpheres World;
#elif BVH
		[ReadOnly] public BvhNode World;
#endif

#if BVH_ITERATIVE
		public NativeArray<IntPtr> NodeWorkingBuffer;
		public NativeArray<Entity> EntityWorkingBuffer;
#if BVH_SIMD
		public NativeArray<float4> VectorWorkingBuffer;
#endif
#endif

#if BUFFERED_MATERIALS
		[ReadOnly] public NativeArray<Material> Material;
#endif

		[WriteOnly] public NativeArray<float4> OutputSamples;
		[WriteOnly] public NativeArray<Diagnostics> OutputDiagnostics;

		public void Execute(int index)
		{
			// ReSharper disable once PossibleLossOfFraction
			float2 coordinates = float2(
				index % Size.x, // column
				index / Size.x  // row
			);

			float4 lastValue = InputSamples[index];

			float3 colorAcc = lastValue.xyz;
			int sampleCount = (int) lastValue.w;

			var rng = new Random(Seed + (uint) index * 0x7383ED49u);
			Diagnostics diagnostics = default;

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
				if (Color(r, 0, rng, workingArea, out float3 sampleColor, ref diagnostics))
#else
				if (Color(r, 0, rng, out float3 sampleColor, ref rayCount))
#endif
				{
					colorAcc += sampleColor;
					sampleCount++;
				}
			}

			OutputSamples[index] = float4(colorAcc, sampleCount);
			OutputDiagnostics[index] = diagnostics;
		}

#if BVH_ITERATIVE
		bool Color(Ray r, uint depth, Random rng, WorkingArea wa, out float3 color, ref Diagnostics diagnostics)
#else
		bool Color(Ray r, uint depth, Random rng, out float3 color, ref uint rayCount)
#endif
		{
			diagnostics.RayCount++;

#if BVH_ITERATIVE
			if (World.Hit(r, 0.001f, float.PositiveInfinity, wa, ref diagnostics, out HitRecord rec))
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
					if (Color(scattered, depth + 1, rng, wa, out float3 scatteredColor, ref diagnostics))
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
		[WriteOnly] public NativeArray<BGRA8> Output;

		public void Execute(int index)
		{
			float4 inputSample = Input[index];

			var realSampleCount = (int) inputSample.w;

			float3 finalColor;
			if (realSampleCount == 0)
				finalColor = NoSamplesColor;
			else
				finalColor = inputSample.xyz / realSampleCount;

			float3 outputColor = finalColor.zyx * 255;

			Output[index] = new BGRA8
			{
				b = (short) outputColor.x,
				g = (short) outputColor.y,
				r = (short) outputColor.z
			};
		}
	}

	struct BGRA8
	{
		public short b, g, r;
#pragma warning disable 649
		public short a;
#pragma warning restore 649
	}
}