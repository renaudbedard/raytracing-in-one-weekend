using System;
using JetBrains.Annotations;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	struct QuadAabb : IDisposable
	{
		// MinX, MaxX, MinY, MaxY, MinZ, MaxZ (4-wide)
		readonly NativeArray<float4> components;

		public unsafe QuadAabb(AxisAlignedBoundingBox box1, AxisAlignedBoundingBox box2, AxisAlignedBoundingBox box3, AxisAlignedBoundingBox box4)
		{
			components = new NativeArray<float4>(6, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

			var fillPtr = (float*) components.GetUnsafePtr();

			var boxes = new NativeArray<AxisAlignedBoundingBox>(4, Allocator.Temp, NativeArrayOptions.UninitializedMemory)
			{
				[0] = box1, [1] = box2, [2] = box3, [3] = box4
			};

			for (int i = 0; i < 4; i++) *fillPtr++ = boxes[i].Min.x;
			for (int i = 0; i < 4; i++) *fillPtr++ = boxes[i].Max.x;

			for (int i = 0; i < 4; i++) *fillPtr++ = boxes[i].Min.y;
			for (int i = 0; i < 4; i++) *fillPtr++ = boxes[i].Max.y;

			for (int i = 0; i < 4; i++) *fillPtr++ = boxes[i].Min.z;
			for (int i = 0; i < 4; i++) *fillPtr++ = boxes[i].Max.z;
			
			boxes.Dispose();
		}

		public unsafe bool4 Hit(Ray r, float4 tMin, float4 tMax)
		{
			bool4 retValue = true;

			var fetchPtr = (float4*) components.GetUnsafeReadOnlyPtr();
			for (int a = 0; a < 3; a++)
			{
				float4 invDirection = 1 / r.Direction[a];
				float4 t0 = (*fetchPtr++ - r.Origin[a]) * invDirection; // Min
				float4 t1 = (*fetchPtr++ - r.Origin[a]) * invDirection; // Max

				bool4 invDirectionNegative = invDirection < 0;
				float4 tt0 = select(t0, t1, invDirectionNegative);
				float4 tt1 = select(t1, t0, invDirectionNegative);

				tMin = select(tt0, tMin, tt0 > tMin);
				tMax = select(tt1, tMax, tt1 < tMax);

				retValue &= tMax > tMin;

				// TODO: is this faster?
				if (!all(retValue)) return false;
			}

			return retValue;
		}

		public void Dispose()
		{
			// ReSharper disable once ImpureMethodCallOnReadonlyValueField
			if (components.IsCreated) components.Dispose();
		}

		public AxisAlignedBoundingBox Enclosure =>
			new AxisAlignedBoundingBox(
				float3(cmin(components[0]), cmin(components[2]), cmin(components[4])),
				float3(cmax(components[1]), cmax(components[3]), cmax(components[5])));
	}
	
	struct AxisAlignedBoundingBox
	{
		public readonly float3 Min, Max;

		public AxisAlignedBoundingBox(float3 min, float3 max)
		{
			Min = min;
			Max = max;
		}

		[Pure]
		public bool Hit(Ray r, float tMin, float tMax)
		{
			// NOTE: I tried a SIMD version of it instead of a loop, and it only ended up slower :(
			for (int a = 0; a < 3; a++)
			{
				float invDirection = 1 / r.Direction[a];
				float t0 = (Min[a] - r.Origin[a]) * invDirection;
				float t1 = (Max[a] - r.Origin[a]) * invDirection;

				if (invDirection < 0)
					Util.Swap(ref t0, ref t1);

				tMin = t0 > tMin ? t0 : tMin;
				tMax = t1 < tMax ? t1 : tMax;

				if (tMax <= tMin)
					return false;
			}

			return true;
		}

		public float3 Center => Min + (Max - Min) / 2;
		public float3 Size => Max - Min;

		public static AxisAlignedBoundingBox Enclose(AxisAlignedBoundingBox lhs, AxisAlignedBoundingBox rhs) =>
			new AxisAlignedBoundingBox(min(lhs.Min, rhs.Min), max(lhs.Max, rhs.Max));
	}
}