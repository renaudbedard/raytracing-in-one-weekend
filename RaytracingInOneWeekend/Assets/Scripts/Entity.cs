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
		public readonly float3 DestinationPosition;
		public readonly float2 TimeRange;
		public readonly Material Material;

		[NativeDisableUnsafePtrRestriction] readonly void* content;

		public Entity(EntityType type, void* contentPointer, RigidTransform transform, Material material)
		{
			Type = type;
			OriginTransform = transform;
			content = contentPointer;
			Material = material;
			TimeRange = float2(0, 1);
			DestinationPosition = OriginTransform.pos;
		}

		public Entity(EntityType type, void* contentPointer, RigidTransform originTransform, float3 destinationPosition, float2 timeRange, Material material)
		{
			Type = type;
			OriginTransform = originTransform;
			DestinationPosition = destinationPosition;
			content = contentPointer;
			TimeRange = timeRange;
			Material = material;
		}

		[Pure]
		public bool Hit(Ray ray, float tMin, float tMax, out HitRecord rec)
		{
			var transformAtTime = new RigidTransform(OriginTransform.rot,
				lerp(OriginTransform.pos, DestinationPosition,
					clamp(unlerp(TimeRange.x, TimeRange.y, ray.Time), 0.0f, 1.0f)));

			RigidTransform inverseTransform = inverse(transformAtTime);

			var entitySpaceRay = new Ray(
				transform(inverseTransform, ray.Origin),
				rotate(inverseTransform, ray.Direction));

			bool hit;
			float distance;
			float3 normal;

			switch (Type)
			{
				case EntityType.Sphere:
					hit = ((Sphere*)content)->Hit(entitySpaceRay, tMin, tMax, out distance, out normal);
					break;

				case EntityType.Rect:
					hit = ((Rect*)content)->Hit(entitySpaceRay, tMin, tMax, out distance, out normal);
					break;

				case EntityType.Box:
					hit = ((Box*)content)->Hit(entitySpaceRay, tMin, tMax, out distance, out normal);
					break;

				default:
					rec = default;
					return false;
			}

			if (!hit)
			{
				rec = default;
				return false;
			}

			rec = new HitRecord(distance, ray.GetPoint(distance), normalize(rotate(transformAtTime, normal)), Material);
			return true;
		}

#if BVH
		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				AxisAlignedBoundingBox bounds;

				switch (Type)
				{
					case EntityType.Sphere: bounds = ((Sphere*)content)->Bounds; break;
					case EntityType.Rect: bounds = ((Rect*)content)->Bounds; break;
					case EntityType.Box: bounds = ((Box*)content)->Bounds; break;
					default: return default;
				}

				float3[] corners = bounds.Corners;

				var minTransform = new RigidTransform(OriginTransform.rot, min(OriginTransform.pos, DestinationPosition));
				var maxTransform = new RigidTransform(OriginTransform.rot, max(OriginTransform.pos, DestinationPosition));

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

#if BVH
	struct EntityBoundsComparer : IComparer<Entity>
	{
		readonly PartitionAxis axis;

		public EntityBoundsComparer(PartitionAxis axis) => this.axis = axis;

		public int Compare(Entity lhs, Entity rhs)
		{
			int axisId = axis.GetAxisId();
			return (int) sign(lhs.Bounds.Center[axisId] - rhs.Bounds.Center[axisId]);
		}
	}
#endif
}
#endif