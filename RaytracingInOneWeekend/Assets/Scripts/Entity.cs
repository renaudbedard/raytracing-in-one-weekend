#if !(AOSOA_SIMD || SOA_SIMD)
using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.Collections.LowLevel.Unsafe;

namespace RaytracerInOneWeekend
{
	unsafe struct Entity
	{
		public readonly EntityType Type;

		[NativeDisableUnsafePtrRestriction] readonly Sphere* sphere;
		[NativeDisableUnsafePtrRestriction] readonly Rect* rect;

		public Entity(Sphere* sphere) : this()
		{
			Type = EntityType.Sphere;
			this.sphere = sphere;
		}
		public Entity(Rect* rect) : this()
		{
			Type = EntityType.Rect;
			this.rect = rect;
		}

		public Sphere* AsSphere => sphere;
		public Rect* AsRect => rect;

		[Pure]
		public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
		{
			switch (Type)
			{
				case EntityType.Sphere: return sphere->Hit(r, tMin, tMax, out rec);
				case EntityType.Rect: return rect->Hit(r, tMin, tMax, out rec);
				default:
					rec = default;
					return false;
			}
		}

		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				switch (Type)
				{
					case EntityType.Sphere: return sphere->Bounds;
					case EntityType.Rect: return rect->Bounds;
					default: return default;
				}
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