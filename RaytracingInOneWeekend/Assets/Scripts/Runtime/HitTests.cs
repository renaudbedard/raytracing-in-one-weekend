using Runtime.EntityTypes;
using Unity.Mathematics;
using static Unity.Mathematics.math;

#if BASIC
using Unity.Collections;
#endif

namespace Runtime
{
	static class HitTests
	{
		public static bool Hit(this AxisAlignedBoundingBox aabb, float3 rayOrigin, float3 rayInvDirection, float tMin, float tMax)
		{
			// optimized algorithm from Roman Wiche
			// https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525

			float3 t0 = (aabb.Min - rayOrigin) * rayInvDirection;
			float3 t1 = (aabb.Max - rayOrigin) * rayInvDirection;

			tMin = max(tMin, cmax(min(t0, t1)));
			tMax = min(tMax, cmin(max(t0, t1)));

			return tMin < tMax;
		}

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
			normal = 0;

			float3 pvec = cross(r.Direction, tri.Data[0]);
			float det = dot(tri.Data[1], pvec);

			// if the determinant is negative the triangle is backfacing
			// if the determinant is close to 0, the ray misses the triangle
			if (det == 0) return false;
			float invDet = 1 / det;

			float3 tvec = r.Origin - tri.Data[2];
			float u = dot(tvec, pvec) * invDet;
			if (u < 0 || u > 1) return false;

			float3 qvec = cross(tvec, tri.Data[1]);
			float v = dot(r.Direction, qvec) * invDet;
			if (v < 0 || u + v > 1) return false;

			distance = dot(tri.Data[0], qvec) * invDet;
			if (distance < tMin || distance > tMax) return false;

			// interpolate normals based on barycentric coordinates
			// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
			float3 barycentricCoords = float3(1 - u - v, u, v);
			normal = mul(tri.Normals, barycentricCoords);
			return true;
		}

#if BASIC
		// iterative entity array hit test
		public static unsafe bool Hit(this NativeArray<Entity> entities, Ray r, float tMin, float tMax, ref RandomSource rng, out HitRecord rec)
		{
			bool anyHit = false;
			rec = new HitRecord(tMax, 0, 0);

			for (var i = 0; i < entities.Length; i++)
			{
				var thisHit = entities[i].Hit(r, tMin, rec.Distance, ref rng, out HitRecord thisRec);
				if (thisHit && (!anyHit || thisRec.Distance < rec.Distance))
				{
					anyHit = true;
					rec = thisRec;
					rec.EntityPtr = (Entity*) entities.GetUnsafeReadOnlyPtr() + i;
				}
			}
			return anyHit;
		}

#else
		public static unsafe bool Hit(this BvhNode n, Ray r, float tMin, float tMax, Material* randomWalkEntryMaterial,
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
					bool thisHit = (n.EntitiesStart + i)->Hit(r, tMin, tMax, randomWalkEntryMaterial, ref rng, out HitRecord thisRec);
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
			bool hitLeft = n.Left->Hit(r, tMin, tMax, randomWalkEntryMaterial, ref rng, ref diagnostics, out HitRecord leftRecord);
			bool hitRight = n.Right->Hit(r, tMin, tMax, randomWalkEntryMaterial, ref rng, ref diagnostics, out HitRecord rightRecord);
#else
			bool hitLeft = n.Left->Hit(r, tMin, tMax, randomWalkEntryMaterial, ref rng, out HitRecord leftRecord);
			bool hitRight = n.Right->Hit(r, tMin, tMax, randomWalkEntryMaterial, ref rng, out HitRecord rightRecord);
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
#endif
	}
}