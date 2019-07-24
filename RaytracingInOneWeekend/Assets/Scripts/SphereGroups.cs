#if SOA_SIMD || AOSOA_SIMD || BVH_SIMD
using Unity.Mathematics;
#endif

#if SOA_SIMD || AOSOA_SIMD
using System;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;
using static Unity.Mathematics.math;
#endif

#if AOSOA_SIMD
using System.Collections.Generic;
using JetBrains.Annotations;
#endif

namespace RaytracerInOneWeekend
{
#if SOA_SIMD
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
		public int Length => CenterX.Length;

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

#elif AOSOA_SIMD
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

#elif BVH_SIMD
	struct Sphere4
	{
		public const int StreamCount = 4;
#pragma warning disable 649
		public float4 CenterX, CenterY, CenterZ, SquaredRadius;
#pragma warning restore 649
	}
#endif
}