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
	[Flags]
	enum PartitionAxis
	{
		None = 0,
		X = 1, 
		Y = 2, 
		Z = 4,
		All = X | Y | Z
	}
	static class PartitionAxisExtensions
	{
		public static int GetAxisId(this PartitionAxis axis)
		{
			switch (axis)
			{
				case PartitionAxis.X: return 0;
				case PartitionAxis.Y: return 1;
				case PartitionAxis.Z: return 2;
			}
			Assert.IsTrue(false, "Multivalued or unset axis");
			return -1;
		}

		public static IEnumerable<PartitionAxis> Enumerate(this PartitionAxis axis)
		{
			if ((axis & PartitionAxis.X) != 0) yield return PartitionAxis.X;
			if ((axis & PartitionAxis.Y) != 0) yield return PartitionAxis.Y;
			if ((axis & PartitionAxis.Z) != 0) yield return PartitionAxis.Z;
		}
	}
	
	struct BvhNode
	{
		public readonly Entity Left;
		public readonly Entity Right;
		public readonly AxisAlignedBoundingBox Bounds;

		public unsafe BvhNode(NativeSlice<Entity> entities, NativeList<BvhNode> nodes)
		{
			var possiblePartitions = PartitionAxis.All;
			var entireBounds = new AxisAlignedBoundingBox(float.MaxValue, float.MinValue);
			foreach (var entity in entities)
			{
				if (entity.GetBounds(out var bounds))
					entireBounds = AxisAlignedBoundingBox.Enclose(entireBounds, bounds);
			}
			
			var biggestPartition = PartitionAxis.None;
			var biggestPartitionSize = float.MinValue;
			foreach (PartitionAxis partition in possiblePartitions.Enumerate())
			{
				float size = entireBounds.Size[partition.GetAxisId()];
				if (size > biggestPartitionSize)
				{
					biggestPartition = partition;
					biggestPartitionSize = size;
				}
			}

			entities.Sort(new EntityBoundsComparer(biggestPartition));

			int n = entities.Length;
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

					var leftNode = new BvhNode(new NativeSlice<Entity>(entities, 0, n / 2), nodes);
					var rightNode = new BvhNode(new NativeSlice<Entity>(entities, n / 2), nodes);

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