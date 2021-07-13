using System;
using JetBrains.Annotations;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine.Assertions;
using static Unity.Mathematics.math;

namespace Runtime
{
	enum EntityType
	{
		None,
		Sphere,
		Rect,
		Box,
		Triangle,
		Mesh
	}

	readonly unsafe struct Entity
	{
		public readonly EntityType Type;
		public readonly bool Moving;
		public readonly RigidTransform OriginTransform;
		public readonly RigidTransform InverseTransform;
		public readonly float3 DestinationOffset;
		public readonly float2 TimeRange;
		public readonly Material* Material;
#if BVH
		// TODO: this is only used in the BVH generation step, and not at runtime
		public readonly AxisAlignedBoundingBox Bounds;
#endif

		[NativeDisableUnsafePtrRestriction] readonly void* content;

		public Entity(EntityType type, void* contentPointer, RigidTransform originTransform, Material* material,
			bool moving = false, float3 destinationOffset = default, float2 timeRange = default) : this()
		{
			Type = type;
			content = contentPointer;
			Moving = moving;
			TimeRange = timeRange;
			OriginTransform = originTransform;
			DestinationOffset = destinationOffset;
			TimeRange = timeRange;
			Material = material;

			if (!moving)
				InverseTransform = inverse(OriginTransform);
			else
				Assert.AreNotEqual(timeRange.x, timeRange.y, "Time range cannot be empty for moving entities.");

#if BVH
			switch (Type)
			{
				case EntityType.Sphere: Bounds = ((Sphere*) content)->Bounds; break;
				case EntityType.Rect: Bounds = ((Rect*) content)->Bounds; break;
				case EntityType.Box: Bounds = ((Box*) content)->Bounds; break;
				case EntityType.Triangle: Bounds = ((Triangle*) content)->Bounds; break;
				default: throw new InvalidOperationException($"Unknown entity type : {Type}");
			}

			float3[] corners = Bounds.Corners;

			float3 destinationPosition = OriginTransform.pos + DestinationOffset;
			var minTransform = new RigidTransform(OriginTransform.rot, min(OriginTransform.pos, destinationPosition));
			var maxTransform = new RigidTransform(OriginTransform.rot, max(OriginTransform.pos, destinationPosition));

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
		public bool Hit(Ray ray, float tMin, float tMax, ref RandomSource rng, out HitRecord rec)
		{
			if (HitInternal(ray, tMin, tMax, ref rng, out float distance, out float3 entityLocalNormal,
				out RigidTransform transformAtTime, out _))
			{
				// TODO: normal is disregarded for isotropic materials
				rec = new HitRecord(distance, ray.GetPoint(distance), normalize(rotate(transformAtTime, entityLocalNormal)));
				return true;
			}

			rec = default;
			return false;
		}

		bool HitInternal(Ray ray, float tMin, float tMax, ref RandomSource rng,
			out float distance, out float3 entitySpaceNormal, out RigidTransform transformAtTime, out Ray entitySpaceRay)
		{
			RigidTransform inverseTransform;

			if (!Moving)
			{
				transformAtTime = OriginTransform;
				inverseTransform = InverseTransform;
			}
			else
			{
				transformAtTime = TransformAtTime(ray.Time);
				inverseTransform = inverse(transformAtTime);
			}

			entitySpaceRay = new Ray(
				transform(inverseTransform, ray.Origin),
				rotate(inverseTransform, ray.Direction));

			if (!HitContent(entitySpaceRay, tMin, tMax, out distance, out entitySpaceNormal))
				return false;

			if (Material->Type == MaterialType.ProbabilisticVolume)
			{
				float entryDistance = distance;
				if (!HitContent(entitySpaceRay, entryDistance + 0.001f, tMax, out float exitDistance, out _))
				{
					// we're inside the boundary, and our first hit was an exit point
					exitDistance = entryDistance;
					entryDistance = 0;
				}

				float distanceInsideBoundary = exitDistance - entryDistance;
				float volumeHitDistance = -(1 / Material->Density) * log(rng.NextFloat());

				if (volumeHitDistance < distanceInsideBoundary)
				{
					distance = entryDistance + volumeHitDistance;
					return true;
				}

				return false;
			}

			return true;
		}

		bool HitContent(Ray r, float tMin, float tMax, out float distance, out float3 normal)
		{
			switch (Type)
			{
				case EntityType.Sphere: return ((Sphere*) content)->Hit(r, tMin, tMax, out distance, out normal);
				case EntityType.Rect: return ((Rect*) content)->Hit(r, tMin, tMax, out distance, out normal);
				case EntityType.Box: return ((Box*) content)->Hit(r, tMin, tMax, out distance, out normal);
				case EntityType.Triangle: return ((Triangle*) content)->Hit(r, tMin, tMax, out distance, out normal);

				default:
					distance = 0;
					normal = default;
					return false;
			}
		}

		public float Pdf(Ray r, ref RandomSource rng)
		{
			if (HitInternal(r, 0.001f, float.PositiveInfinity, ref rng, out float distance,
				out float3 entitySpaceNormal, out _, out Ray entitySpaceRay))
			{
				switch (Type)
				{
					case EntityType.Rect:
						return ((Rect*) content)->Pdf(entitySpaceRay.Direction, distance, entitySpaceNormal);

					case EntityType.Sphere:
						return ((Sphere*) content)->Pdf(entitySpaceRay.Origin);

					default: throw new NotImplementedException();
				}
			}

			return 0;
		}

		public float3 RandomPoint(float time, ref RandomSource rng)
		{
			float3 localPoint;
			switch (Type)
			{
				case EntityType.Rect: localPoint = ((Rect*) content)->RandomPoint(ref rng); break;
				case EntityType.Sphere: localPoint = ((Sphere*) content)->RandomPoint(ref rng); break;
				default: throw new NotImplementedException();
			}

			return transform(TransformAtTime(time), localPoint);
		}

		public RigidTransform TransformAtTime(float t) =>
			new RigidTransform(OriginTransform.rot,
				OriginTransform.pos +
				DestinationOffset * clamp(unlerp(TimeRange.x, TimeRange.y, t), 0.0f, 1.0f));
	}
}