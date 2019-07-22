using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

#if UNITY_SOA
using Unity.Collections.Experimental;
#endif

namespace RaytracerInOneWeekend
{
#if MANUAL_AOSOA
	unsafe struct AosoaSpheres : IDisposable
	{
		public enum Streams { CenterX, CenterY, CenterZ, SquaredRadius }
		public const int StreamCount = 4;

		static readonly int SimdLength = sizeof(float4) / sizeof(float);

		public readonly int Length, BlockCount;

		NativeArray<float4> dataBuffer;

		// NOTE: radius and material are not stored as a stream since we don't need them during iteration
		public NativeArray<float> Radius;
#if BUFFERED_MATERIALS
		public NativeArray<int> MaterialIndex;
#else
		public NativeArray<Material> Material;
#endif

		public AosoaSpheres(int length)
		{
			Length = length;
			BlockCount = (int) ceil(length / (float) SimdLength);

			dataBuffer = new NativeArray<float4>(BlockCount * StreamCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			Radius = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

#if BUFFERED_MATERIALS
			MaterialIndex = new NativeArray<int>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
#else
			Material = new NativeArray<Material>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
#endif
			float4* lastBlockPointer = (float4*) dataBuffer.GetUnsafePtr() + (BlockCount - 1) * StreamCount;
			lastBlockPointer[(int) Streams.CenterX] = float.MaxValue;
			lastBlockPointer[(int) Streams.CenterY] = float.MaxValue;
			lastBlockPointer[(int) Streams.CenterZ] = float.MaxValue;
			lastBlockPointer[(int) Streams.SquaredRadius] = 0;
		}

		public float4* ReadOnlyDataPointer => (float4*) dataBuffer.GetUnsafeReadOnlyPtr();
		public float4* GetReadOnlyBlockPointer(int blockIndex) => ReadOnlyDataPointer + blockIndex * StreamCount;

		public void SetElement(int i, float3 center, float radius)
		{
			GetOffsets(i, out int blockIndex, out int lane);

			float4* blockPointer = (float4*) dataBuffer.GetUnsafePtr() + blockIndex * StreamCount;
			blockPointer[(int) Streams.CenterX][lane] = center.x;
			blockPointer[(int) Streams.CenterY][lane] = center.y;
			blockPointer[(int) Streams.CenterZ][lane] = center.z;
			blockPointer[(int) Streams.SquaredRadius][lane] = radius * radius;

			Radius[i] = radius;
		}

		public void GetOffsets(int i, out int blockIndex, out int lane)
		{
			blockIndex = i / SimdLength;
			lane = i % SimdLength;
		}

		public void Dispose()
		{
			if (dataBuffer.IsCreated) dataBuffer.Dispose();
			if (Radius.IsCreated) Radius.Dispose();
#if BUFFERED_MATERIALS
			if (MaterialIndex.IsCreated) MaterialIndex.Dispose();
#else
			if (Material.IsCreated) Material.Dispose();
#endif
		}
	}

#elif MANUAL_SOA
	struct SoaSpheres : IDisposable
	{
		public readonly int BlockCount;

		public NativeArray<float> CenterX, CenterY, CenterZ;
		public NativeArray<float> SquaredRadius, Radius;
#if BUFFERED_MATERIALS
		public NativeArray<int> MaterialIndex;
#else
		public NativeArray<Material> Material;
#endif
		public int Count => CenterX.Length;

		public SoaSpheres(int length)
		{
			int dataLength = length;
			BlockCount = (int) ceil(length / 4.0);
			length = BlockCount * 4;

			CenterX = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			CenterY = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			CenterZ = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			SquaredRadius = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			Radius = new NativeArray<float>(dataLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
#if BUFFERED_MATERIALS
			MaterialIndex = new NativeArray<int>(dataLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
#else
			Material = new NativeArray<Material>(dataLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
#endif
			for (int i = dataLength; i < length; i++)
			{
				CenterX[i] = CenterY[i] = CenterZ[i] = float.PositiveInfinity;
				SquaredRadius[i] = 0;
			}
		}

		public void SetElement(int i, float3 center, float radius)
		{
			CenterX[i] = center.x;
			CenterY[i] = center.y;
			CenterZ[i] = center.z;
			Radius[i] = radius;
			SquaredRadius[i] = radius * radius;
		}

		public void Dispose()
		{
			if (CenterX.IsCreated) CenterX.Dispose();
			if (CenterY.IsCreated) CenterY.Dispose();
			if (CenterZ.IsCreated) CenterZ.Dispose();
			if (SquaredRadius.IsCreated) SquaredRadius.Dispose();
			if (Radius.IsCreated) Radius.Dispose();
#if BUFFERED_MATERIALS
			if (MaterialIndex.IsCreated) MaterialIndex.Dispose();
#else
			if (Material.IsCreated) Material.Dispose();
#endif
		}
	}

#else // AOS (or possibly Unity AOS)
	struct Sphere
	{
		public readonly float3 Center;
		public readonly float SquaredRadius;
		public readonly float Radius;
#if BUFFERED_MATERIALS || UNITY_SOA
		public readonly int MaterialIndex;
#else
		public readonly Material Material;
#endif

#if BUFFERED_MATERIALS || UNITY_SOA
		public Sphere(float3 center, float radius, int materialIndex)
#else
		public Sphere(float3 center, float radius, Material material)
#endif
		{
			Center = center;
			Radius = radius;
			SquaredRadius = radius * radius;
#if BUFFERED_MATERIALS || UNITY_SOA
			MaterialIndex = materialIndex;
#else
			Material = material;
#endif
		}
		
		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				float absRadius = abs(Radius);
				return new AxisAlignedBoundingBox(Center - absRadius, Center + absRadius);
			}
		}		
	}
#endif
}