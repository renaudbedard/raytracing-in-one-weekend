#if !(MANUAL_AOSOA || MANUAL_SOA)
using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.Collections.LowLevel.Unsafe;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	enum EntityType
	{
		None,
		Sphere
	}

	unsafe struct Entity
	{
		public readonly EntityType Type;

		[NativeDisableUnsafePtrRestriction] readonly Sphere* sphere;

		public Entity(Sphere* sphere) : this()
		{
			Type = EntityType.Sphere;
			this.sphere = sphere;
		}

		[Pure]
		public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
		{
			switch (Type)
			{
				case EntityType.Sphere: return sphere->Hit(r, tMin, tMax, out rec);
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