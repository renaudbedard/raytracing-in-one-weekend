using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Assertions;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	struct BvhNode
	{
		public readonly Entity Left;
		public readonly Entity Right;
		public readonly AxisAlignedBoundingBox Bounds;

		public unsafe BvhNode(NativeSlice<Entity> entities, NativeList<BvhNode> nodes, Random rng)
		{
			int n = entities.Length;
			int axis = rng.NextInt(0, 3);
			entities.Sort(new EntityBoundsComparer(axis));

			switch (n)
			{
				case 1:
					Left = Right = entities[0];
					break;

				case 2:
					Left = entities[0];
					Right = entities[1];
					break;

				default:
					var nodesPointer = (BvhNode*) nodes.GetUnsafePtr();

					var leftNode = new BvhNode(new NativeSlice<Entity>(entities, 0, n / 2), nodes, rng);
					var rightNode = new BvhNode(new NativeSlice<Entity>(entities, n / 2), nodes, rng);

					nodes.Add(leftNode);
					Left = new Entity(nodesPointer + (nodes.Length - 1));

					nodes.Add(rightNode);
					Right = new Entity(nodesPointer + (nodes.Length - 1));
					break;
			}

			bool hasLeftBounds = Left.GetBounds(out AxisAlignedBoundingBox leftBounds);
			Assert.IsTrue(hasLeftBounds, $"{Left} has no bounds");

			bool hasRightBounds = Right.GetBounds(out AxisAlignedBoundingBox rightBounds);
			Assert.IsTrue(hasRightBounds, $"{Right} has no bounds");

			Bounds = AxisAlignedBoundingBox.Enclose(leftBounds, rightBounds);
		}
	}
}