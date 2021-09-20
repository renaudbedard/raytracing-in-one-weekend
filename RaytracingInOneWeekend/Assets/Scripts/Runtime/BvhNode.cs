using Unity.Collections.LowLevel.Unsafe;

namespace Runtime
{
	readonly unsafe struct BvhNode
	{
		public readonly AxisAlignedBoundingBox Bounds;
		[NativeDisableUnsafePtrRestriction] public readonly BvhNode* Left, Right;
		public readonly Entity* EntitiesStart;
		public readonly int EntityCount;

		public BvhNode(AxisAlignedBoundingBox bounds, Entity* entitiesStart, int entityCount, BvhNode* left, BvhNode* right)
		{
			Bounds = bounds;
			EntitiesStart = entitiesStart;
			EntityCount = entityCount;
			Left = left;
			Right = right;
		}

		public bool IsLeaf => EntitiesStart != null;
	}
}