using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;
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
			Assert.IsTrue(lengthsq(leftBounds.Size) > 0, $"{Left} has zero bounds");

			bool hasRightBounds = Right.GetBounds(out AxisAlignedBoundingBox rightBounds);
			Assert.IsTrue(hasRightBounds, $"{Right} has no bounds");
			Assert.IsTrue(lengthsq(rightBounds.Size) > 0, $"{Right} has zero bounds");

			Bounds = AxisAlignedBoundingBox.Enclose(leftBounds, rightBounds);
		}

		public IEnumerable<ValueTuple<AxisAlignedBoundingBox, int>> GetAllSubBounds(int depth = 0)
		{
			yield return (Bounds, depth);

			if (Left.Type == EntityType.BvhNode)
				foreach ((var bounds, int subdepth) in Left.AsNode.GetAllSubBounds(depth + 1))
					yield return (bounds, subdepth);
			else if (Left.GetBounds(out var leftBounds))
				yield return (leftBounds, depth + 1);

			if (Right.Type == EntityType.BvhNode)
				foreach ((var bounds, int subdepth) in Right.AsNode.GetAllSubBounds(depth + 1))
					yield return (bounds, subdepth);
			else if (Right.GetBounds(out var rightBounds))
				yield return (rightBounds, depth + 1);
		}
	}
}