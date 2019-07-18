using Unity.Mathematics;
using static Unity.Mathematics.math;

#if !MANUAL_AOSOA
using Unity.Collections;
#endif

#if MANUAL_SOA || MANUAL_AOSOA
using Unity.Collections.LowLevel.Unsafe;
#endif

#if UNITY_SOA
using Unity.Collections.Experimental;
#endif

namespace RaytracerInOneWeekend
{
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
				pCenterX++; pCenterY++; pCenterZ++;
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
			float closestRadius = spheres.Radius[closestId];
#else
			spheres.GetOffsets(closestId, out int blockIndex, out int lane);
			blockCursor = spheres.GetReadOnlyBlockPointer(blockIndex);

			float3 closestCenter = float3(
				blockCursor[(int)AosoaSpheres.Streams.CenterX][lane],
				blockCursor[(int)AosoaSpheres.Streams.CenterY][lane],
				blockCursor[(int)AosoaSpheres.Streams.CenterZ][lane]);
			float closestRadius = spheres.Radius[closestId];
#endif

#if BUFFERED_MATERIALS
			int closestMaterial = spheres.MaterialIndex[closestId];
#else
			Material closestMaterial = spheres.Material[closestId];
#endif

			float3 point = r.GetPoint(minDistance);
			rec = new HitRecord(minDistance, point, (point - closestCenter) / closestRadius, closestMaterial);
			return true;
		}

#else
#if UNITY_SOA
		public static bool Hit(this NativeArrayFullSOA<Sphere> spheres, Ray r, float tMin, float tMax, out HitRecord rec)
#else
#if BVH
		public static bool Hit(this NativeArray<BvhNode> entities, Ray r, float tMin, float tMax, out HitRecord rec)
#else
		public static bool Hit(this NativeArray<Entity> entities, Ray r, float tMin, float tMax, out HitRecord rec)
#endif
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
			for (var i = 0; i < entities.Length; i++)
			{
				if (entities[i].Hit(r, tMin, rec.Distance, out HitRecord thisRec))
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
			float radius = s.Radius;
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
					rec = new HitRecord(t, point, (point - center) / radius, material);
					return true;
				}

				t = (-b + sqrtDiscriminant) / a;
				if (t < tMax && t > tMin)
				{
					float3 point = r.GetPoint(t);
					rec = new HitRecord(t, point, (point - center) / radius, material);
					return true;
				}
			}

			rec = default;
			return false;
		}

		public static bool Hit(this BvhNode n, Ray r, float tMin, float tMax, out HitRecord rec)
		{
			rec = default;
			if (n.Bounds.Hit(r, tMin, tMax))
			{
				bool hitLeft = n.Left.Hit(r, tMin, tMax, out HitRecord leftRecord);
				bool hitRight = n.Right.Hit(r, tMin, tMax, out HitRecord rightRecord);

				if (hitLeft && hitRight)
				{
					rec = leftRecord.Distance < rightRecord.Distance ? leftRecord : rightRecord;
					return true;
				}
				if (hitLeft)
				{
					rec = leftRecord;
					return true;
				}
				if (hitRight)
				{
					rec = rightRecord;
					return true;
				}
				return false;
			}

			return false;
		}

		public static bool GetBounds(this NativeArray<Entity> entities, out AxisAlignedBoundingBox enclosingAabb)
		{
			bool anyBox = false;
			enclosingAabb = default;
			for (int i = 0; i < entities.Length; i++)
			{
				anyBox |= entities[i].GetBounds(out AxisAlignedBoundingBox aabb);
				enclosingAabb = i == 0 ? aabb : AxisAlignedBoundingBox.Enclose(enclosingAabb, aabb);
			}
			return anyBox;
		}

		public static bool GetBounds(this BvhNode node, out AxisAlignedBoundingBox bounds)
		{
			bounds = node.Bounds;
			return true;
		}

		public static bool GetBounds(this Sphere s, out AxisAlignedBoundingBox box)
		{
			box = new AxisAlignedBoundingBox(s.Center - s.Radius, s.Center + s.Radius);
			return true;
		}
#endif
	}
}