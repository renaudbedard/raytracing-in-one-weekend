#if !(AOSOA_SIMD || SOA_SIMD)
using JetBrains.Annotations;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;
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

		[NativeDisableUnsafePtrRestriction] readonly void* content;

		public Entity(EntityType type, void* contentPointer, RigidTransform transform, Material material,
			float3 destinationOffset = default, float2 timeRange = default)
		{
			Type = type;
			OriginTransform = transform;
			content = contentPointer;
			TimeRange = timeRange;
			DestinationOffset = destinationOffset;
			TimeRange = timeRange;
			Material = material;
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

#if BVH
		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				AxisAlignedBoundingBox bounds;

				switch (Type)
				{
					case EntityType.Sphere:
						bounds = ((Sphere*) content)->Bounds;
						break;
					case EntityType.Rect:
						bounds = ((Rect*) content)->Bounds;
						break;
					case EntityType.Box:
						bounds = ((Box*) content)->Bounds;
						break;
					default: return default;
				}

				float3[] corners = bounds.Corners;

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

				return new AxisAlignedBoundingBox(minimum, maximum);
			}
		}
#endif
	}
}
#endif