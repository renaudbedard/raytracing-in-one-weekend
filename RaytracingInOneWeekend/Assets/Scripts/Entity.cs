using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.Collections.LowLevel.Unsafe;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	enum EntityType
	{
		None,
		Sphere,
#if BVH
		BvhNode,
#endif
	}

	unsafe struct Entity
	{
		public readonly EntityType Type;

		[NativeDisableUnsafePtrRestriction] readonly Sphere* sphere;
#if BVH
		[NativeDisableUnsafePtrRestriction] readonly BvhNode* bvhNode;
#endif

		public Entity(Sphere* sphere) : this()
		{
			Type = EntityType.Sphere;
			this.sphere = sphere;
		}
#if BVH
		public Entity(BvhNode* bvhNode) : this()
		{
			Type = EntityType.BvhNode;
			this.bvhNode = bvhNode;
		}

		public BvhNode AsNode => *bvhNode;
#endif

		[Pure]
		public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
		{
			switch (Type)
			{
				case EntityType.Sphere: return sphere->Hit(r, tMin, tMax, out rec);
#if BVH && !BVH_ITERATIVE
				case EntityType.BvhNode: return bvhNode->Hit(r, tMin, tMax, out rec);
#endif
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
#if BVH
					case EntityType.BvhNode: return bvhNode->Bounds;
#endif
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