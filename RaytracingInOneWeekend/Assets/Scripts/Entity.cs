#if !(AOSOA_SIMD || SOA_SIMD)
using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	unsafe struct Entity
	{
		public readonly EntityType Type;
		public readonly RigidTransform Transform, InverseTransform;
		public readonly Material Material;

		// TODO: reimplement motion

		[NativeDisableUnsafePtrRestriction] readonly void* content;

		public Entity(EntityType type, void* contentPointer, RigidTransform transform, Material material) : this()
		{
			Type = type;
			Transform = transform;
			InverseTransform = inverse(transform);
			content = contentPointer;
			Material = material;
		}

		[Pure]
		public bool Hit(Ray ray, float tMin, float tMax, out HitRecord rec)
		{
			var entitySpaceRay = new Ray(
				transform(InverseTransform, ray.Origin),
				rotate(InverseTransform, ray.Direction));

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

			rec = new HitRecord(distance, ray.GetPoint(distance), rotate(Transform, normal), Material);
			return true;
		}

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

				return new AxisAlignedBoundingBox(transform(Transform, bounds.Min), transform(Transform, bounds.Max));
			}
		}
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