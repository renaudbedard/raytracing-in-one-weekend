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
#if FULL_DIAGNOSTICS && BVH_ITERATIVE
	struct Diagnostics
	{
		public float RayCount;
		public float BoundsHitCount;
		public float CandidateCount;
#pragma warning disable 649
		public float Padding;
#pragma warning restore 649
	}
#else
	struct Diagnostics
	{
		public float RayCount;
	}
#endif

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	unsafe struct AccumulateJob : IJobParallelFor
	{
#if BVH_ITERATIVE
		public struct WorkingArea
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
		[ReadOnly] public Camera Camera;
		[ReadOnly] public uint SampleCount;
		[ReadOnly] public int TraceDepth;
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
		[ReadOnly]
		[NativeDisableUnsafePtrRestriction]
		public BvhNode* World;
#endif

		[ReadOnly] public PerlinData PerlinData;

#if BVH_ITERATIVE
		[ReadOnly] public int NodeCount;
		[ReadOnly] public int EntityCount;
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

			// big prime stolen from Unity's random class
			var rng = new Random((Seed * 0x8C4CA03Fu) ^ (uint) (index * 0x7383ED49u));
			Diagnostics diagnostics = default;

#if BVH_ITERATIVE
			BvhNode** nodes = stackalloc BvhNode*[NodeCount];
			Entity* entities = stackalloc Entity[EntityCount];
#if BVH_SIMD
			int maxVectorWorkingSizePerEntity = sizeof(Sphere4) / sizeof(float4);
			var entityGroupCount = (int) ceil(EntityCount / 4.0f);
			float4* vectors = stackalloc float4[maxVectorWorkingSizePerEntity * entityGroupCount];
#endif
			var workingArea = new WorkingArea
			{
				Nodes = nodes,
				Entities = entities,
#if BVH_SIMD
				Vectors = vectors
#endif
			};
#endif

			float3* emissionStack = stackalloc float3[TraceDepth];
			float3* attenuationStack = stackalloc float3[TraceDepth];

			for (int s = 0; s < SampleCount; s++)
			{
				float2 normalizedCoordinates = (coordinates + rng.NextFloat2()) / Size; // (u, v)
				Ray r = Camera.GetRay(normalizedCoordinates, rng);

				if (Color(r, rng, emissionStack, attenuationStack,
#if BVH_ITERATIVE
					workingArea,
#endif
					out float3 sampleColor, ref diagnostics))
				{
					colorAcc += sampleColor;
					sampleCount++;
				}
			}

			OutputSamples[index] = float4(colorAcc, sampleCount);
			OutputDiagnostics[index] = diagnostics;
		}

		bool Color(Ray r, Random rng, float3* emissionStack, float3* attenuationStack,
#if BVH_ITERATIVE
			WorkingArea wa,
#endif
			out float3 color, ref Diagnostics diagnostics)
		{
			float3* emissionCursor = emissionStack;
			float3* attenuationCursor = attenuationStack;

			int depth = 0;
			for (; depth < TraceDepth; depth++)
			{
#if BVH
				bool hit = World->Hit(
#else
				bool hit = World.Hit(
#endif
					r, 0, float.PositiveInfinity,
#if BVH_ITERATIVE
					wa,
#endif
#if FULL_DIAGNOSTICS
					ref diagnostics,
#endif
					out HitRecord rec);

				diagnostics.RayCount++;

				if (hit)
				{
					*emissionCursor++ = rec.Material.Emit(rec.Point, rec.Normal, PerlinData);
					bool didScatter = rec.Material.Scatter(r, rec, rng, PerlinData, out float3 attenuation, out r);
					*attenuationCursor++ = attenuation;
					if (didScatter) r = r.OffsetTowards(dot(r.Direction, rec.Normal) > 0 ? rec.Normal : -rec.Normal);
					else break;

					// scatter debugging
					//rec.Material.Scatter(r, rec, rng, PerlinData, out float3 _, out r);
					//*emissionCursor++ = normalize(r.Direction) * 0.5f + 0.5f;
					////*emissionCursor++ = normalize(rec.Normal) * 0.5f + 0.5f;
					//*attenuationCursor++ = 0;
					// break;
				}
				else
				{
					// sample the sky color
					float3 unitDirection = normalize(r.Direction);
					float t = 0.5f * (unitDirection.y + 1);
					*emissionCursor++ = lerp(SkyBottomColor, SkyTopColor, t);
					*attenuationCursor++ = 0;
					break;
				}
			}

			color = 0;

			if (depth == TraceDepth)
				return false;

			// attenuate colors from the tail of the hit stack to the head
			while (emissionCursor != emissionStack)
			{
				emissionCursor--;
				attenuationCursor--;
				color = color * *attenuationCursor + *emissionCursor;
			}
			return true;
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct CombineJob : IJobParallelFor
	{
		static readonly float3 NoSamplesColor = new float3(1, 0, 1);

		[ReadOnly] public NativeArray<float4> Input;
		[WriteOnly] public NativeArray<RGBA32> Output;

		public void Execute(int index)
		{
			float4 inputSample = Input[index];

			var realSampleCount = (int) inputSample.w;

			float3 finalColor;
			if (realSampleCount == 0)
				finalColor = NoSamplesColor;
			else
				finalColor = inputSample.xyz / realSampleCount;

			// TODO: tone-mapping
			float3 outputColor = saturate(finalColor.xyz.LinearToGamma()) * 255;

			// scatter debugging
			// float3 outputColor = normalize(inputSample.xyz) * 255;

			Output[index] = new RGBA32
			{
				r = (byte) outputColor.x,
				g = (byte) outputColor.y,
				b = (byte) outputColor.z,
				a = 255
			};
		}
	}
}