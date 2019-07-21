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
		BvhNode
	}

	unsafe struct Entity
	{
		public readonly EntityType Type;

		[NativeDisableUnsafePtrRestriction] readonly Sphere* sphere;
		[NativeDisableUnsafePtrRestriction] readonly BvhNode* bvhNode;

		public Entity(Sphere* sphere) : this()
		{
			Type = EntityType.Sphere;
			this.sphere = sphere;
		}
		public Entity(BvhNode* bvhNode) : this()
		{
			Type = EntityType.BvhNode;
			this.bvhNode = bvhNode;
		}

		public BvhNode AsNode => *bvhNode;

		[Pure]
		public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
		{
			switch (Type)
			{
				case EntityType.Sphere: return sphere->Hit(r, tMin, tMax, out rec);
				case EntityType.BvhNode: return bvhNode->Hit(r, tMin, tMax, out rec);

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
					case EntityType.BvhNode: return bvhNode->Bounds;
					default: return default;
				}
			}
		}
	}

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
}