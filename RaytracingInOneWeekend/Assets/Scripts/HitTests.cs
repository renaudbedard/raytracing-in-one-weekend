using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	static class HitTests
	{
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

			if (r.Direction.z >= 0) return false;
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

#if BASIC
		// iterative entity array hit test
		public static bool Hit(this NativeArray<Entity> entities, Ray r, float tMin, float tMax, ref RandomSource rng, out HitRecord rec)
		{
			bool hitAnything = false;
			rec = new HitRecord(tMax, 0, 0, default);

			for (var i = 0; i < entities.Length; i++)
			{
				if (entities[i].Hit(r, tMin, rec.Distance, ref rng, out HitRecord thisRec))
				{
					hitAnything = true;
					rec = thisRec;
				}
			}

			return hitAnything;
		}

#elif BVH_RECURSIVE
		public static unsafe bool Hit(this BvhNode n, NativeArray<Entity> entities, Ray r, float tMin, float tMax,
			ref RandomSource rng,
#if FULL_DIAGNOSTICS
			ref Diagnostics diagnostics,
#endif
			out HitRecord rec)
		{
			rec = default;
			float3 rayInvDirection = rcp(r.Direction);

			if (!n.Bounds.Hit(r.Origin, rayInvDirection, tMin, tMax))
				return false;

#if FULL_DIAGNOSTICS
			diagnostics.BoundsHitCount++;
#endif

			if (n.IsLeaf)
			{
#if FULL_DIAGNOSTICS
				diagnostics.CandidateCount++;
#endif
				return entities[n.EntityId].Hit(r, tMin, tMax, ref rng, out rec);
			}

#if FULL_DIAGNOSTICS
			bool hitLeft = n.Left->Hit(entities, r, tMin, tMax, ref rng, ref diagnostics, out HitRecord leftRecord);
			bool hitRight = n.Right->Hit(entities, r, tMin, tMax, ref rng, ref diagnostics, out HitRecord rightRecord);
#else
			bool hitLeft = n.Left->Hit(entities, r, tMin, tMax, ref rng, out HitRecord leftRecord);
			bool hitRight = n.Right->Hit(entities, r, tMin, tMax, ref rng, out HitRecord rightRecord);
#endif

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
		public static unsafe bool Hit(this BvhNode node, NativeArray<Entity> entities, Ray r, float tMin, float tMax,
			ref RandomSource rng, AccumulateJob.WorkingArea workingArea,
#if FULL_DIAGNOSTICS
			ref Diagnostics diagnostics,
#endif
			out HitRecord rec)
		{
			int candidateCount = 0, nodeStackHeight = 1;
			BvhNode** nodeStackTail = workingArea.Nodes;
			Entity* candidateListTail = workingArea.Entities - 1, candidateListHead = workingArea.Entities;
			float3 rayInvDirection = rcp(r.Direction);

			*nodeStackTail = &node;

			while (nodeStackHeight > 0)
			{
				BvhNode* nodePtr = *nodeStackTail--;
				nodeStackHeight--;

				if (!nodePtr->Bounds.Hit(r.Origin, rayInvDirection, tMin, tMax))
					continue;

				if (nodePtr->IsLeaf)
				{
					*++candidateListTail = entities[nodePtr->EntityId];
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