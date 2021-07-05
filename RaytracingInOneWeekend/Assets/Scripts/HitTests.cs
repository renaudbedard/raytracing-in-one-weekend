using Unity.Mathematics;
using UnityEngine.Assertions;
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

		// ray direction is assumed to be normalized
		public static bool Hit(this Triangle tri, Ray r, float tMin, float tMax, out float distance, out float3 normal)
		{
			// from "Fast, Minimum Storage Ray/Triangle Intersection" (Möller & Trumbore)
			// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
			distance = 0;
			normal = tri.Normal;

			float3 pvec = cross(r.Direction, tri.AC);
			float det = dot(tri.AB, pvec);

			// if the determinant is negative the triangle is backfacing
			// if the determinant is close to 0, the ray misses the triangle
			if (det == 0) return false;

			float invDet = 1 / det;

			float3 tvec = r.Origin - tri.A;
			float u = dot(tvec, pvec) * invDet;
			if (u < 0 || u > 1) return false;

			float3 qvec = cross(tvec, tri.AB);
			float v = dot(r.Direction, qvec) * invDet;
			if (v < 0 || u + v > 1) return false;

			distance = dot(tri.AC, qvec) * invDet;
			if (distance < tMin || distance > tMax) return false;

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
		public static unsafe bool Hit(this BvhNode n, Ray r, float tMin, float tMax,
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
				diagnostics.CandidateCount += n.EntityCount;
#endif
				bool anyHit = false;
				for (int i = 0; i < n.EntityCount; i++)
				{
					bool thisHit = (n.EntitiesStart + i)->Hit(r, tMin, tMax, ref rng, out HitRecord thisRec);
					if (thisHit && (!anyHit || thisRec.Distance < rec.Distance))
					{
						anyHit = true;
						rec = thisRec;
						rec.EntityPtr = n.EntitiesStart + i;
					}
				}
				return anyHit;
			}

#if FULL_DIAGNOSTICS
			bool hitLeft = n.Left->Hit(r, tMin, tMax, ref rng, ref diagnostics, out HitRecord leftRecord);
			bool hitRight = n.Right->Hit(r, tMin, tMax, ref rng, ref diagnostics, out HitRecord rightRecord);
#else
			bool hitLeft = n.Left->Hit(r, tMin, tMax, ref rng, out HitRecord leftRecord);
			bool hitRight = n.Right->Hit(r, tMin, tMax, ref rng, out HitRecord rightRecord);
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
		public static unsafe bool Hit(this BvhNode node, Ray r, float tMin, float tMax,
			ref RandomSource rng,
#if FULL_DIAGNOSTICS
			ref Diagnostics diagnostics,
#endif
			out HitRecord rec)
		{
			rec = default;

			var np = stackalloc BvhNode*[1] { &node };
			var nodesToTraverse = new PointerBlock<BvhNode>(np, 1) { ChainLength = 1 };

			var ep = stackalloc Entity*[1];
			var hitCandidates = new PointerBlock<Entity>(ep, 1);

			float3 rayInvDirection = rcp(r.Direction);

			while (nodesToTraverse.ChainLength > 0)
			{
				BvhNode* nodePtr = nodesToTraverse.Pop();

				if (!nodePtr->Bounds.Hit(r.Origin, rayInvDirection, tMin, tMax))
					continue;

#if FULL_DIAGNOSTICS
				diagnostics.BoundsHitCount++;
#endif

				if (nodePtr->IsLeaf)
				{
					int entityCount = nodePtr->EntityCount;
					Entity* entityPtr = nodePtr->EntitiesStart;

					for (int i = 0; i < entityCount; i++)
					{
						// TODO: We should be able to preallocate for entityCount
						if (!hitCandidates.TryPush(entityPtr, out var parent))
						{
							var pb = stackalloc PointerBlock<Entity>[1];
							var p = stackalloc Entity*[parent->Capacity * 2];
							*pb = new PointerBlock<Entity>(p, parent->Capacity * 2);
							parent->NextBlock = pb;
							hitCandidates.TryPush(entityPtr, out _);
						}
					}

#if FULL_DIAGNOSTICS
					diagnostics.CandidateCount += entityCount;
#endif
				}
				else
				{
					// TODO: We should be able to preallocate for 2
					if (!nodesToTraverse.TryPush(nodePtr->Left, out var parent))
					{
						var pb = stackalloc PointerBlock<BvhNode>[1];
						var p = stackalloc BvhNode*[parent->Capacity * 2];
						*pb = new PointerBlock<BvhNode>(p, parent->Capacity * 2);
						parent->NextBlock = pb;
						nodesToTraverse.TryPush(nodePtr->Left, out _);
					}
					if (!nodesToTraverse.TryPush(nodePtr->Right, out parent))
					{
						var pb = stackalloc PointerBlock<BvhNode>[1];
						var p = stackalloc BvhNode*[parent->Capacity * 2];
						*pb = new PointerBlock<BvhNode>(p, parent->Capacity * 2);
						parent->NextBlock = pb;
						nodesToTraverse.TryPush(nodePtr->Right, out _);
					}
				}
			}

			// iterative candidate tests
			bool anyHit = false;
			while (hitCandidates.ChainLength > 0)
			{
				var hitCandidate = hitCandidates.Pop();
				bool thisHit = hitCandidate->Hit(r, tMin, tMax, ref rng, out HitRecord thisRec);
				if (thisHit && (!anyHit || thisRec.Distance < rec.Distance))
				{
					anyHit = true;
					rec = thisRec;
					rec.EntityPtr = hitCandidate;
				}
			}
			if (anyHit)
				return true;

			return false;
		}
#endif
	}
}