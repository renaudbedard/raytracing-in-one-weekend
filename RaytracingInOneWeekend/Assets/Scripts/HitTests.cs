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
	// TODO: do we even need tMin and tMax?

	static class HitTests
	{
#if !(AOSOA_SIMD || SOA_SIMD)
		// single sphere hit test
		public static bool Hit(this Sphere s, Ray r, float tMin, float tMax, out float distance, out float3 normal)
		{
			float squaredRadius = s.SquaredRadius;
			float radius = s.Radius;

			float3 oc = r.Origin;
			float a = dot(r.Direction, r.Direction);
			float b = dot(oc, r.Direction);
			float c = dot(oc, oc) - squaredRadius;
			float discriminant = b * b - a * c;

			if (discriminant > 0)
			{
				float sqrtDiscriminant = sqrt(discriminant);

				// TODO: this breaks when a == 0
				float t = (-b - sqrtDiscriminant) / a;
				if (t < tMax && t > tMin)
				{
					distance = t;
					normal = r.GetPoint(t) / radius;
					return true;
				}

				// TODO: this breaks when a == 0
				t = (-b + sqrtDiscriminant) / a;
				if (t < tMax && t > tMin)
				{
					distance = t;
					normal = r.GetPoint(t) / radius;
					return true;
				}
			}

			distance = 0;
			normal = 0;
			return false;
		}

		public static bool Hit(this Rect rect, Ray r, float tMin, float tMax, out float distance, out float3 normal)
		{
			distance = 0;
			normal = 0;

			if (r.Direction.z.AlmostEquals(0)) return false;
			float t = -r.Origin.z / r.Direction.z;
			if (t < tMin || t > tMax) return false;

			float2 xy = r.Origin.xy + t * r.Direction.xy;
			bool4 test = bool4(xy < rect.From, xy > rect.To);
			if (any(test)) return false;

			distance = t;
			normal = float3(0, 0, 1);
			return true;
		}

		// ray direction is assumed to be normalized
		public static bool Hit(this Box box, Ray r, float tMin, float tMax, out float distance, out float3 normal)
		{
			// offset origin by tMin
			r = new Ray(r.Origin + r.Direction * tMin, r.Direction, r.Time);

			distance = 0;
			normal = 0;

			// from "A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering" (Majercik et al.)
			// http://jcgt.org/published/0007/03/04/
			float3 rayDirection = r.Direction;
			float winding = cmax(abs(r.Origin) * box.InverseExtents) < 1 ? -1 : 1;
			float3 sgn = -sign(rayDirection);
			float3 distanceToPlane = (box.Extents * winding * sgn - r.Origin) / rayDirection;

			bool3 test = distanceToPlane >= 0 & bool3(
				all(abs(r.Origin.yz + rayDirection.yz * distanceToPlane.x) < box.Extents.yz),
				all(abs(r.Origin.zx + rayDirection.zx * distanceToPlane.y) < box.Extents.zx),
				all(abs(r.Origin.xy + rayDirection.xy * distanceToPlane.z) < box.Extents.xy));

			sgn = test.x ? float3(sgn.x, 0, 0) : test.y ? float3(0, sgn.y, 0) : float3(0, 0, test.z ? sgn.z : 0);
			bool3 nonZero = sgn != 0;
			if (!any(nonZero)) return false;

			distance = nonZero.x ? distanceToPlane.x : nonZero.y ? distanceToPlane.y : distanceToPlane.z;
			distance += tMin;

			if (distance > tMax) return false;

			normal = sgn;
			return true;
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
		// TODO: this is fully broken

#if SOA_SIMD
		public static unsafe bool Hit(this SoaSpheres s, Ray r, float tMin, float tMax, out HitRecord rec)
#elif AOSOA_SIMD
		public static unsafe bool Hit(this AosoaSpheres s, Ray r, float tMin, float tMax, out HitRecord rec)
#endif
		{
#if SOA_SIMD
			float4* centerFromX = s.PtrCenterFromX, centerFromY = s.PtrCenterFromY, centerFromZ = s.PtrCenterFromZ;
			float4* centerToX = s.PtrCenterToX, centerToY = s.PtrCenterToY, centerToZ = s.PtrCenterToZ;
			float4* fromTime = s.PtrFromTime, toTime = s.PtrToTime;
			float4* sqRadius = s.PtrSqRadius;
#elif AOSOA_SIMD
			float4* blockCursor = s.ReadOnlyDataPointer;
#endif
			rec = new HitRecord(tMax, 0, 0, default);
			float4 a = dot(r.Direction, r.Direction);
			int4 curId = int4(0, 1, 2, 3), hitId = -1;
			float4 hitT = tMax;
			int count = s.BlockCount;

			for (int i = 0; i < count; i++)
			{
#if AOSOA_SIMD
				float4* fromTime = blockCursor + (int) AosoaSpheres.Streams.FromTime,
					toTime = blockCursor + (int) AosoaSpheres.Streams.ToTime,
					centerFromX = blockCursor + (int) AosoaSpheres.Streams.CenterFromX,
					centerToX = blockCursor + (int) AosoaSpheres.Streams.CenterToX,
					centerFromY = blockCursor + (int) AosoaSpheres.Streams.CenterFromY,
					centerToY = blockCursor + (int) AosoaSpheres.Streams.CenterToY,
					centerFromZ = blockCursor + (int) AosoaSpheres.Streams.CenterFromZ,
					centerToZ = blockCursor + (int) AosoaSpheres.Streams.CenterToZ,
					sqRadius = blockCursor + (int) AosoaSpheres.Streams.SquaredRadius;
#endif
				float4 timeStep = saturate(unlerp(*fromTime, *toTime, r.Time));

				float4 centerX = lerp(*centerFromX, *centerToX, timeStep),
					centerY = lerp(*centerFromY, *centerToY, timeStep),
					centerZ = lerp(*centerFromZ, *centerToZ, timeStep);

				float4 ocX = r.Origin.x - centerX,
					ocY = r.Origin.y - centerY,
					ocZ = r.Origin.z - centerZ;

				float4 b = ocX * r.Direction.x + ocY * r.Direction.y + ocZ * r.Direction.z;
				float4 c = ocX * ocX + ocY * ocY + ocZ * ocZ - *sqRadius;
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
				++centerFromX; ++centerFromY; ++centerFromZ;
				++centerToX; ++centerToY; ++centerToZ;
				++fromTime; ++toTime;
				++sqRadius;
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
			float3 closestCenterFrom = float3(s.CenterFromX[closestId], s.CenterFromY[closestId], s.CenterFromZ[closestId]);
			float3 closestCenterTo = float3(s.CenterToX[closestId], s.CenterToY[closestId], s.CenterToZ[closestId]);
			float closestTimeStep = saturate(unlerp(s.FromTime[closestId], s.ToTime[closestId], r.Time));
			float closestRadius = s.Radius[closestId];
#elif AOSOA_SIMD
			s.GetOffsets(closestId, out int blockIndex, out int lane);
			blockCursor = s.GetReadOnlyBlockPointer(blockIndex);

			float3 closestCenterFrom = float3(
				blockCursor[(int)AosoaSpheres.Streams.CenterFromX][lane],
				blockCursor[(int)AosoaSpheres.Streams.CenterFromY][lane],
				blockCursor[(int)AosoaSpheres.Streams.CenterFromZ][lane]);
			float3 closestCenterTo = float3(
				blockCursor[(int)AosoaSpheres.Streams.CenterToX][lane],
				blockCursor[(int)AosoaSpheres.Streams.CenterToY][lane],
				blockCursor[(int)AosoaSpheres.Streams.CenterToZ][lane]);
			float closestTimeStep = saturate(unlerp(
				blockCursor[(int)AosoaSpheres.Streams.FromTime][lane],
				blockCursor[(int)AosoaSpheres.Streams.ToTime][lane], r.Time));
			float closestRadius = s.Radius[closestId];
#endif
			float3 closestCenter = lerp(closestCenterFrom, closestCenterTo, closestTimeStep);

			Material closestMaterial = s.Material[closestId];

			float3 point = r.GetPoint(minDistance);
			rec = new HitRecord(minDistance, point, (point - closestCenter) / closestRadius, closestMaterial);
			return true;
		}

#elif BVH_RECURSIVE
		public static unsafe bool Hit(this BvhNode n, Ray r, float tMin, float tMax,
#if FULL_DIAGNOSTICS
			ref AccumulateJob.Diagnostics diagnostics,
#endif
			out HitRecord rec)
		{
			rec = default;

			if (!n.Bounds.Hit(r, tMin, tMax))
				return false;

#if FULL_DIAGNOSTICS
			diagnostics.BoundsHitCount++;
#endif

			if (n.IsLeaf)
			{
#if FULL_DIAGNOSTICS
				diagnostics.CandidateCount++;
#endif
				return n.Content.Hit(r, tMin, tMax, out rec);
			}

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
		public static unsafe bool Hit(this BvhNode n, Ray r, float tMin, float tMax, ref Random rng,
			AccumulateJob.WorkingArea wa,
#if FULL_DIAGNOSTICS
			ref Diagnostics diagnostics,
#endif
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
#if FULL_DIAGNOSTICS
				diagnostics.BoundsHitCount++;
#endif
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
#if FULL_DIAGNOSTICS
			diagnostics.CandidateCount = candidateCount;
#endif
			if (candidateCount == 0)
			{
				rec = default;
				return false;
			}

#if BVH_SIMD
			// TODO: this is fully broken
			// skip SIMD codepath if there's only one
			if (candidateCount == 1)
				return candidateListHead->Hit(r, tMin, tMax, out rec);

			var simdSpheresHead = (Sphere4*) wa.Vectors;
			int simdBlockCount = (int) ceil(candidateCount / 4.0f);

			float4 a = dot(r.Direction, r.Direction);
			int4 curId = int4(0, 1, 2, 3), hitId = -1;
			float4 hitT = tMax;

			Sphere4* blockCursor = simdSpheresHead;
			Entity* candidateCursor = candidateListHead;
			int candidateIndex = 0;
			for (int i = 0; i < simdBlockCount; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					if (candidateIndex < candidateCount)
					{
						Sphere* sphereData = candidateCursor->AsSphere;
						float3 center = sphereData->Center(r.Time);
						blockCursor->CenterX[j] = center.x;
						blockCursor->CenterY[j] = center.y;
						blockCursor->CenterZ[j] = center.z;
						blockCursor->SquaredRadius[j] = sphereData->SquaredRadius;
						++candidateCursor;
						++candidateIndex;
					}
					else
					{
						blockCursor->CenterX[j] = float.MaxValue;
						blockCursor->CenterY[j] = float.MaxValue;
						blockCursor->CenterZ[j] = float.MaxValue;
						blockCursor->SquaredRadius[j] = 0;
					}
				}

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

			rec = new HitRecord(minDistance, point,
				(point - closestSphere->Center(r.Time)) / closestSphere->Radius,
				closestSphere->Material);

			return true;

#else
			// iterative candidate tests (non-SIMD)
			bool anyHit = candidateListHead->Hit(r, tMin, tMax, ref rng, out rec);
			for (int i = 1; i < candidateCount; i++)
			{
				bool thisHit = candidateListHead[i].Hit(r, tMin, tMax, ref rng, out HitRecord thisRec);
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