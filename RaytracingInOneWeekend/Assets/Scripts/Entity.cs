using System;
using JetBrains.Annotations;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;
#if BVH
using System.Collections.Generic;

#endif

namespace RaytracerInOneWeekend
{
	unsafe struct Entity
	{
		public readonly EntityType Type;
		public readonly RigidTransform OriginTransform;
		public readonly float3 DestinationOffset;
		public readonly float2 TimeRange;
		public readonly Material Material;
#if BVH
		// TODO: this is only used in the BVH generation step, and not at runtime
		public readonly AxisAlignedBoundingBox Bounds;
#endif

		[NativeDisableUnsafePtrRestriction] readonly void* content;

		public Entity(EntityType type, void* contentPointer, RigidTransform originTransform, Material material,
			float3 destinationOffset = default, float2 timeRange = default)
		{
			Type = type;
			OriginTransform = originTransform;
			content = contentPointer;
			TimeRange = timeRange;
			DestinationOffset = destinationOffset;
			TimeRange = timeRange;
			Material = material;

#if BVH
			switch (Type)
			{
				case EntityType.Sphere:
					Bounds = ((Sphere*) content)->Bounds;
					break;
				case EntityType.Rect:
					Bounds = ((Rect*) content)->Bounds;
					break;
				case EntityType.Box:
					Bounds = ((Box*) content)->Bounds;
					break;
				default: throw new InvalidOperationException($"Unknown entity type : {Type}");
			}

			float3[] corners = Bounds.Corners;

			float3 destinationPosition = OriginTransform.pos + DestinationOffset;
			var minTransform =
				new RigidTransform(OriginTransform.rot, min(OriginTransform.pos, destinationPosition));
			var maxTransform =
				new RigidTransform(OriginTransform.rot, max(OriginTransform.pos, destinationPosition));

			var minimum = new float3(float.PositiveInfinity);
			var maximum = new float3(float.NegativeInfinity);
			foreach (float3 c in corners)
			{
				minimum = min(minimum, transform(minTransform, c));
				maximum = max(maximum, transform(maxTransform, c));
			}

			Bounds = new AxisAlignedBoundingBox(minimum, maximum);
#endif
		}

		[Pure]
		public bool Hit(Ray ray, float tMin, float tMax, ref Random rng, out HitRecord rec)
		{
			var transformAtTime = new RigidTransform(OriginTransform.rot,
				OriginTransform.pos +
				DestinationOffset * clamp(unlerp(TimeRange.x, TimeRange.y, ray.Time), 0.0f, 1.0f));

			RigidTransform inverseTransform = inverse(transformAtTime);

			var entitySpaceRay = new Ray(
				transform(inverseTransform, ray.Origin),
				rotate(inverseTransform, ray.Direction));

			if (!HitContent(entitySpaceRay, tMin, tMax, out float distance, out float3 normal))
			{
				rec = default;
				return false;
			}

			if (Material.Type == MaterialType.Isotropic)
			{
				float entryDistance = distance;
				if (!HitContent(entitySpaceRay, entryDistance + 0.001f, tMax, out float exitDistance, out _))
				{
					// we're inside the boundary, and our first hit was an exit point
					exitDistance = entryDistance;
					entryDistance = 0;
				}

				float distanceInsideBoundary = exitDistance - entryDistance;
				float volumeHitDistance = -(1 / Material.Parameter) * log(rng.NextFloat());

				if (volumeHitDistance < distanceInsideBoundary)
				{
					distance = entryDistance + volumeHitDistance;
					rec = new HitRecord(distance, ray.GetPoint(distance), default, Material);
					return true;
				}

				rec = default;
				return false;
			}

			rec = new HitRecord(distance, ray.GetPoint(distance), normalize(rotate(transformAtTime, normal)), Material);
			return true;
		}

		bool HitContent(Ray r, float tMin, float tMax, out float distance, out float3 normal)
		{
			switch (Type)
			{
				case EntityType.Sphere: return ((Sphere*) content)->Hit(r, tMin, tMax, out distance, out normal);
				case EntityType.Rect: return ((Rect*) content)->Hit(r, tMin, tMax, out distance, out normal);
				case EntityType.Box: return ((Box*) content)->Hit(r, tMin, tMax, out distance, out normal);

				default:
					distance = 0;
					normal = default;
					return false;
			}
		}

		public float PdfValue(Ray r, ref Random rng)
		{
			if (Hit(r, 0.001f, float.PositiveInfinity, ref rng, out HitRecord rec))
			{
				switch (Type)
				{
					case EntityType.Rect:
						float area = ((Rect*) content)->Area;
						float distanceSquared = rec.Distance * rec.Distance;
						float cosine = abs(dot(r.Direction, rec.Normal));
						return distanceSquared / (cosine * area);

					default:
						// TODO
						return 0;
				}
			}

			return 0;
		}

		public float3 RandomPoint(float time, ref Random rng)
		{
			var transformAtTime = new RigidTransform(OriginTransform.rot,
				OriginTransform.pos +
				DestinationOffset * clamp(unlerp(TimeRange.x, TimeRange.y, time), 0.0f, 1.0f));

			switch (Type)
			{
				case EntityType.Rect:
					var rect = (Rect*) content;
					float3 localPoint = float3(rng.NextFloat2(rect->From, rect->To), 0);
					return transform(transformAtTime, localPoint);

				default:
					// TODO
					return 0;
			}
		}
	}
}