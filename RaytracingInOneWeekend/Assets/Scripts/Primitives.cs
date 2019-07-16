using System;
using Unity.Collections;
using Unity.Collections.Experimental;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

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

		public NativeArray<Material> Materials;

		public AosoaSpheres(int length)
		{
			Length = length;

			BlockCount = (int) ceil(length / (float) SimdLength);
			dataBuffer = new NativeArray<float4>(BlockCount * StreamCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

			Materials = new NativeArray<Material>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

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
		}

		public void GetOffsets(int i, out int blockIndex, out int lane)
		{
			blockIndex = i / SimdLength;
			lane = i % SimdLength;
		}

		public void Dispose()
		{
			if (dataBuffer.IsCreated) dataBuffer.Dispose();
			if (Materials.IsCreated) Materials.Dispose();
		}
	}
#elif MANUAL_SOA
	struct SoaSpheres : IDisposable
	{
		public readonly int BlockCount;

		public NativeArray<float> CenterX, CenterY, CenterZ;
		public NativeArray<float> SquaredRadius;
#if BUFFERED_MATERIALS
		public NativeArray<ushort> MaterialIndex;
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
#if BUFFERED_MATERIALS
			MaterialIndex = new NativeArray<ushort>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
#else
			Material = new NativeArray<Material>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
#endif
			for (int i = dataLength; i < length; i++)
			{
				CenterX[i] = CenterY[i] = CenterZ[i] = float.PositiveInfinity;
				SquaredRadius[i] = 0;
			}
		}

		public void Dispose()
		{
			if (CenterX.IsCreated) CenterX.Dispose();
			if (CenterY.IsCreated) CenterY.Dispose();
			if (CenterZ.IsCreated) CenterZ.Dispose();
			if (SquaredRadius.IsCreated) SquaredRadius.Dispose();
#if BUFFERED_MATERIALS
			if (MaterialIndex.IsCreated) MaterialIndex.Dispose();
#else
			if (Material.IsCreated) Material.Dispose();
#endif
		}
	}
#else
	enum PrimitiveType
	{
		None,
		Sphere
	}

	struct Primitive
	{
		public readonly PrimitiveType Type;

		[ReadOnly] readonly NativeSlice<Sphere> sphere;

		// TODO: do we need a public accessor to the underlying primitive?

		public Primitive(NativeSlice<Sphere> sphere)
		{
			UnityEngine.Assertions.Assert.IsTrue(sphere.Length == 1, "Primitive cannot be multi-valued");
			Type = PrimitiveType.Sphere;
			this.sphere = sphere;
		}

		public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
		{
			switch (Type)
			{
				case PrimitiveType.Sphere:
					return sphere[0].Hit(r, tMin, tMax, out rec);

				default:
					rec = default;
					return false;
			}
		}
	}

	struct Sphere
	{
		public readonly float3 Center;
		public readonly float SquaredRadius;
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
			SquaredRadius = radius * radius;
#if BUFFERED_MATERIALS || UNITY_SOA
			MaterialIndex = materialIndex;
#else
			Material = material;
#endif
		}
	}
#endif

	static class WorldExtensions
	{
#if MANUAL_AOSOA || MANUAL_SOA
#if MANUAL_SOA
		public static unsafe bool Hit(this SoaSpheres spheres, Ray r, float tMin, float tMax, out HitRecord rec)
#else
		public static unsafe bool Hit(this AosoaSpheres spheres, Ray r, float tMin, float tMax, out HitRecord rec)
#endif
		{
#if MANUAL_SOA
			float4* pCenterX = (float4*) spheres.CenterX.GetUnsafeReadOnlyPtr(),
				pCenterY = (float4*) spheres.CenterY.GetUnsafeReadOnlyPtr(),
				pCenterZ = (float4*) spheres.CenterZ.GetUnsafeReadOnlyPtr(),
				pSqRadius = (float4*) spheres.SquaredRadius.GetUnsafeReadOnlyPtr();
#else
			float4* blockCursor = spheres.ReadOnlyDataPointer;
#endif

			rec = new HitRecord(tMax, 0, 0, default);
			float4 a = dot(r.Direction, r.Direction);
			int4 curId = int4(0, 1, 2, 3), hitId = -1;
			float4 hitT = tMax;
			int count = spheres.BlockCount;

			for (int i = 0; i < count; i++)
			{
#if MANUAL_SOA
				float4 centerX = *pCenterX, centerY = *pCenterY, centerZ = *pCenterZ, sqRadius = *pSqRadius;
#else
				float4 centerX = *(blockCursor + (int) AosoaSpheres.Streams.CenterX),
					centerY = *(blockCursor + (int) AosoaSpheres.Streams.CenterY),
					centerZ = *(blockCursor + (int) AosoaSpheres.Streams.CenterZ),
					sqRadius = *(blockCursor + (int) AosoaSpheres.Streams.SquaredRadius);
#endif

				float4 ocX = r.Origin.x - centerX,
					ocY = r.Origin.y - centerY,
					ocZ = r.Origin.z - centerZ;

				float4 b = ocX * r.Direction.x + ocY * r.Direction.y + ocZ * r.Direction.z;
				float4 c = ocX * ocX + ocY * ocY + ocZ * ocZ - sqRadius;
				float4 discriminant = b * b - a * c;

				bool4 discriminantTest = discriminant > 0;

				if (any(discriminantTest))
				{
					float4 sqrtDiscriminant = sqrt(discriminant);

					float4 t0 = (-b - sqrtDiscriminant) / a;
					float4 t1 = (-b + sqrtDiscriminant) / a;

					float4 t = select(t1, t0, t0 > tMin);
					bool4 mask = discriminantTest & t > tMin & t < hitT;

					hitId = select(hitId, curId, mask);
					hitT = select(hitT, t, mask);
				}

				curId += 4;

#if MANUAL_SOA
				pCenterX++;
				pCenterY++;
				pCenterZ++;
				pSqRadius++;
#else
				blockCursor += AosoaSpheres.StreamCount;
#endif
			}

			if (all(hitId == -1))
				return false;

			float minDistance = cmin(hitT);
			int laneMask = bitmask(hitT == minDistance);
			int firstLane = tzcnt(laneMask);
			int closestId = hitId[firstLane];

#if MANUAL_SOA
			float3 closestCenter = float3(spheres.CenterX[closestId],
				spheres.CenterY[closestId],
				spheres.CenterZ[closestId]);
			float closestRadius = sqrt(spheres.SquaredRadius[closestId]);
			Material closestMaterial = spheres.Material[closestId];
#else
			spheres.GetOffsets(closestId, out int blockIndex, out int lane);
			blockCursor = spheres.GetReadOnlyBlockPointer(blockIndex);

			float3 closestCenter = float3(
				blockCursor[(int)AosoaSpheres.Streams.CenterX][lane],
				blockCursor[(int)AosoaSpheres.Streams.CenterY][lane],
				blockCursor[(int)AosoaSpheres.Streams.CenterZ][lane]);
			float closestRadius = sqrt(blockCursor[(int)AosoaSpheres.Streams.SquaredRadius][lane]);
			Material closestMaterial = spheres.Materials[closestId];
#endif

			float3 point = r.GetPoint(minDistance);
			rec = new HitRecord(minDistance, point, (point - closestCenter) / closestRadius, closestMaterial);
			return true;
		}

#else
#if UNITY_SOA
		public static bool Hit(this NativeArrayFullSOA<Sphere> spheres, Ray r, float tMin, float tMax, out HitRecord rec)
#else
		public static bool Hit(this NativeArray<Primitive> primitives, Ray r, float tMin, float tMax, out HitRecord rec)
#endif
		{
			bool hitAnything = false;
			rec = new HitRecord(tMax, 0, 0, default);

#if UNITY_SOA
			for (var i = 0; i < spheres.Length; i++)
			{
				Sphere sphere = spheres[i];
				if (sphere.Hit(r, tMin, rec.Distance, out HitRecord thisRec))
#else
			for (var i = 0; i < primitives.Length; i++)
			{
				Primitive primitive = primitives[i];
				if (primitive.Hit(r, tMin, rec.Distance, out HitRecord thisRec))
#endif
				{
					hitAnything = true;
					rec = thisRec;
				}
			}

			return hitAnything;
		}

		public static bool Hit(this Sphere s, Ray r, float tMin, float tMax, out HitRecord rec)
		{
			float3 center = s.Center;
			float squaredRadius = s.SquaredRadius;
#if BUFFERED_MATERIALS || UNITY_SOA
			int material = s.MaterialIndex;
#else
			Material material = s.Material;
#endif
			float3 oc = r.Origin - center;
			float a = dot(r.Direction, r.Direction);
			float b = dot(oc, r.Direction);
			float c = dot(oc, oc) - squaredRadius;
			float discriminant = b * b - a * c;

			if (discriminant > 0)
			{
				float sqrtDiscriminant = sqrt(discriminant);
				float t = (-b - sqrtDiscriminant) / a;
				if (t < tMax && t > tMin)
				{
					float3 point = r.GetPoint(t);
					rec = new HitRecord(t, point, (point - center) / sqrt(squaredRadius), material);
					return true;
				}

				t = (-b + sqrtDiscriminant) / a;
				if (t < tMax && t > tMin)
				{
					float3 point = r.GetPoint(t);
					rec = new HitRecord(t, point, (point - center) / sqrt(squaredRadius), material);
					return true;
				}
			}

			rec = default;
			return false;
		}
#endif
	}
}