using Unity.Mathematics;
using static Unity.Mathematics.math;

#if !BVH && !SOA_SIMD && !AOSOA_SIMD
using Unity.Collections;
#endif

#if SOA_SIMD || AOSOA_SIMD
using Unity.Collections.LowLevel.Unsafe;
#endif

namespace RaytracerInOneWeekend
{
	// TODO: lots of code duplication between SoA/AoSoA and BVH SIMD

	static class HitTests
	{
#if !(AOSOA_SIMD || SOA_SIMD)
		// single sphere hit test
		public static bool Hit(this Sphere s, Ray r, float tMin, float tMax, out HitRecord rec)
		{
			float3 center = s.Center;
			float squaredRadius = s.SquaredRadius;
			float radius = s.Radius;
#if BUFFERED_MATERIALS
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
#endif

#if BASIC
		// iterative entity array hit test
		public static bool Hit(this NativeArray<Entity> entities, Ray r, float tMin, float tMax, out HitRecord rec)
		{
			bool hitAnything = false;
			rec = new HitRecord(tMax, 0, 0, default);

			for (var i = 0; i < entities.Length; i++)
			{
				if (entities[i].Hit(r, tMin, rec.Distance, out HitRecord thisRec))
				{
					hitAnything = true;
					rec = thisRec;
				}
			}

			return hitAnything;
		}

#elif AOSOA_SIMD || SOA_SIMD
#if SOA_SIMD
		public static unsafe bool Hit(this SoaSpheres spheres, Ray r, float tMin, float tMax, out HitRecord rec)
#elif AOSOA_SIMD
		public static unsafe bool Hit(this AosoaSpheres spheres, Ray r, float tMin, float tMax, out HitRecord rec)
#endif
		{
#if SOA_SIMD
			float4* pCenterX = (float4*) spheres.CenterX.GetUnsafeReadOnlyPtr(),
				pCenterY = (float4*) spheres.CenterY.GetUnsafeReadOnlyPtr(),
				pCenterZ = (float4*) spheres.CenterZ.GetUnsafeReadOnlyPtr(),
				pSqRadius = (float4*) spheres.SquaredRadius.GetUnsafeReadOnlyPtr();
#elif AOSOA_SIMD
			float4* blockCursor = spheres.ReadOnlyDataPointer;
#endif

			rec = new HitRecord(tMax, 0, 0, default);
			float4 a = dot(r.Direction, r.Direction);
			int4 curId = int4(0, 1, 2, 3), hitId = -1;
			float4 hitT = tMax;
			int count = spheres.BlockCount;

			for (int i = 0; i < count; i++)
			{
#if SOA_SIMD
				float4 centerX = *pCenterX, centerY = *pCenterY, centerZ = *pCenterZ, sqRadius = *pSqRadius;
#elif AOSOA_SIMD
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

#if SOA_SIMD
				pCenterX++; pCenterY++; pCenterZ++;
				pSqRadius++;
#elif AOSOA_SIMD
				blockCursor += AosoaSpheres.StreamCount;
#endif
			}

			if (all(hitId == -1))
				return false;

			float minDistance = cmin(hitT);
			int laneMask = bitmask(hitT == minDistance);
			int firstLane = tzcnt(laneMask);
			int closestId = hitId[firstLane];

#if SOA_SIMD
			float3 closestCenter = float3(spheres.CenterX[closestId],
				spheres.CenterY[closestId],
				spheres.CenterZ[closestId]);
			float closestRadius = spheres.Radius[closestId];
#elif AOSOA_SIMD
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

#elif BVH_RECURSIVE
		public static unsafe bool Hit(this BvhNode n, Ray r, float tMin, float tMax, out HitRecord rec)
		{
			rec = default;

			if (!n.Bounds.Hit(r, tMin, tMax))
				return false;

			if (n.IsLeaf)
				return n.Content.Hit(r, tMin, tMax, out rec);

			bool hitLeft = n.Left->Hit(r, tMin, tMax, out HitRecord leftRecord);
			bool hitRight = n.Right->Hit(r, tMin, tMax, out HitRecord rightRecord);

			if (!hitLeft && !hitRight)
				return false;

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

			rec = rightRecord;
			return true;
		}

#elif BVH_ITERATIVE
		public static unsafe bool Hit(this BvhNode n, Ray r, float tMin, float tMax, AccumulateJob.WorkingArea wa,
			out HitRecord rec)
		{
			int candidateCount = 0, nodeStackHeight = 1;
			BvhNode** nodeStackTail = wa.Nodes;
			Entity* candidateListTail = wa.Entities - 1, candidateListHead = wa.Entities;

			*nodeStackTail = &n;

			while (nodeStackHeight > 0)
			{
				BvhNode* nodePtr = *nodeStackTail--;
				nodeStackHeight--;

				if (!nodePtr->Bounds.Hit(r, tMin, tMax))
					continue;

				if (nodePtr->IsLeaf)
				{
					*++candidateListTail = nodePtr->Content;
					candidateCount++;
				}
				else
				{
					*++nodeStackTail = nodePtr->Left;
					*++nodeStackTail = nodePtr->Right;
					nodeStackHeight += 2;
				}
			}

			if (candidateCount == 0)
			{
				rec = default;
				return false;
			}

			// TODO: visualization for candidate count per pixel

#if BVH_SIMD
			// skip SIMD codepath if there's only one
			if (candidateCount == 1)
				return candidateListHead->Hit(r, tMin, tMax, out rec);

			var simdSpheresHead = (Sphere4*) wa.Vectors;
			int simdBlockCount = (int) ceil(candidateCount / 4.0f);

			if (candidateCount % 4 != 0)
			{
				Sphere4* lastBlock = simdSpheresHead + (simdBlockCount - 1);
				lastBlock->CenterX = float.MaxValue;
				lastBlock->CenterY = float.MaxValue;
				lastBlock->CenterZ = float.MaxValue;
				lastBlock->SquaredRadius = 0;
			}

			Sphere4* blockCursor = simdSpheresHead;
			for (int i = 0; i < simdBlockCount; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					int candidateIndex = i * 4 + j;
					if (candidateIndex < candidateCount)
					{
						var sphereData = candidateListHead[i * 4 + j].AsSphere;
						blockCursor->CenterX[j] = sphereData->Center.x;
						blockCursor->CenterY[j] = sphereData->Center.y;
						blockCursor->CenterZ[j] = sphereData->Center.z;
						blockCursor->SquaredRadius[j] = sphereData->SquaredRadius;
					}
				}
				++blockCursor;
			}

			float4 a = dot(r.Direction, r.Direction);
			int4 curId = int4(0, 1, 2, 3), hitId = -1;
			float4 hitT = tMax;

			blockCursor = simdSpheresHead;
			for (int i = 0; i < simdBlockCount; i++)
			{
				float4 ocX = r.Origin.x - blockCursor->CenterX,
					ocY = r.Origin.y - blockCursor->CenterY,
					ocZ = r.Origin.z - blockCursor->CenterZ;

				float4 b = ocX * r.Direction.x + ocY * r.Direction.y + ocZ * r.Direction.z;
				float4 c = ocX * ocX + ocY * ocY + ocZ * ocZ - blockCursor->SquaredRadius;
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
				++blockCursor;
			}

			if (all(hitId == -1))
			{
				rec = default;
				return false;
			}

			float minDistance = cmin(hitT);
			int laneMask = bitmask(hitT == minDistance);
			int firstLane = tzcnt(laneMask);
			int closestId = hitId[firstLane];
			Sphere* closestSphere = candidateListHead[closestId].AsSphere;

			float3 point = r.GetPoint(minDistance);
			rec = new HitRecord(minDistance, point, (point - closestSphere->Center) / closestSphere->Radius,
#if BUFFERED_MATERIALS
				closestSphere->MaterialIndex);
#else
				closestSphere->Material);
#endif
			return true;

#else
			// iterative candidate tests (non-SIMD)
			bool anyHit = candidateListHead->Hit(r, tMin, tMax, out rec);
			for (int i = 1; i < candidateCount; i++)
			{
				bool thisHit = candidateListHead[i].Hit(r, tMin, tMax, out HitRecord thisRec);
				if (thisHit && (!anyHit || thisRec.Distance < rec.Distance))
				{
					anyHit = true;
					rec = thisRec;
				}
			}
			if (anyHit)
				return true;

			rec = default;
			return false;
#endif
		}
#endif
	}
}