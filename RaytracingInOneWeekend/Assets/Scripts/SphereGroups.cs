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
	unsafe struct SoaSpheres : IDisposable
	{
		public readonly int BlockCount;

		public NativeArray<float> CenterFromX, CenterFromY, CenterFromZ;
		public NativeArray<float> CenterToX, CenterToY, CenterToZ;
		public NativeArray<float> FromTime, ToTime;
		public NativeArray<float> SquaredRadius, Radius;
		public NativeArray<Material> Material;

		[NativeDisableUnsafePtrRestriction]
		public readonly float4* PtrCenterFromX, PtrCenterFromY, PtrCenterFromZ,
			PtrCenterToX, PtrCenterToY, PtrCenterToZ,
			PtrFromTime, PtrToTime,
			PtrSqRadius;

		public int Length => CenterFromX.Length;

		public SoaSpheres(int length)
		{
			int dataLength = length;
			BlockCount = (int) ceil(length / 4.0);
			length = BlockCount * 4;

			CenterFromX = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			CenterFromY = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			CenterFromZ = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			CenterToX = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			CenterToY = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			CenterToZ = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			FromTime = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			ToTime = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			SquaredRadius = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

			Radius = new NativeArray<float>(dataLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			Material = new NativeArray<Material>(dataLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

			for (int i = dataLength; i < length; i++)
			{
				CenterFromX[i] = CenterFromY[i] = CenterFromZ[i] = float.PositiveInfinity;
				CenterToX[i] = CenterToY[i] = CenterToZ[i] = float.PositiveInfinity;
				FromTime[i] = ToTime[i] = 0;
				SquaredRadius[i] = 0;
			}

			// precache pointers
			PtrCenterFromX = (float4*) CenterFromX.GetUnsafeReadOnlyPtr();
			PtrCenterFromY = (float4*) CenterFromY.GetUnsafeReadOnlyPtr();
			PtrCenterFromZ = (float4*) CenterFromZ.GetUnsafeReadOnlyPtr();
			PtrCenterToX = (float4*) CenterToX.GetUnsafeReadOnlyPtr();
			PtrCenterToY = (float4*) CenterToY.GetUnsafeReadOnlyPtr();
			PtrCenterToZ = (float4*) CenterToZ.GetUnsafeReadOnlyPtr();
			PtrFromTime = (float4*) FromTime.GetUnsafeReadOnlyPtr();
			PtrToTime = (float4*) ToTime.GetUnsafeReadOnlyPtr();
			PtrSqRadius = (float4*) SquaredRadius.GetUnsafeReadOnlyPtr();
		}

		public void SetElement(int i, float3 fromCenter, float3 toCenter, float t0, float t1, float radius)
		{
			CenterFromX[i] = fromCenter.x; CenterFromY[i] = fromCenter.y; CenterFromZ[i] = fromCenter.z;
			CenterToX[i] = toCenter.x; CenterToY[i] = toCenter.y; CenterToZ[i] = toCenter.z;
			FromTime[i] = t0; ToTime[i] = t1;
			Radius[i] = radius;
			SquaredRadius[i] = radius * radius;
		}

		public void Dispose()
		{
			CenterFromX.SafeDispose(); CenterFromY.SafeDispose(); CenterFromZ.SafeDispose();
			CenterToX.SafeDispose(); CenterToY.SafeDispose(); CenterToZ.SafeDispose();
			FromTime.SafeDispose(); ToTime.SafeDispose();
			SquaredRadius.SafeDispose();
			Radius.SafeDispose();
			Material.SafeDispose();
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
		public NativeArray<Material> Material;

		public AosoaSpheres(int length)
		{
			Length = length;
			BlockCount = (int) ceil(length / (float) SimdLength);

			dataBuffer = new NativeArray<float4>(BlockCount * StreamCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			Radius = new NativeArray<float>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			Material = new NativeArray<Material>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

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
			dataBuffer.SafeDispose();
			Radius.SafeDispose();
			Material.SafeDispose();
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