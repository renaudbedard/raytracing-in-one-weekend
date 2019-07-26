#if BVH
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Assertions;

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

	unsafe struct BvhNode
	{
		public readonly AxisAlignedBoundingBox Bounds;
		public readonly bool IsLeaf;
		[NativeDisableUnsafePtrRestriction] public readonly BvhNode* Left, Right;
		public readonly Entity Content;

		public static int GetNodeCount(NativeArray<Entity> entities)
		{
			using (var tempNodes = new NativeList<BvhNode>(Allocator.Temp))
			{
				tempNodes.Add(new BvhNode(entities, tempNodes));
				return tempNodes.Length;
			}
		}

		public BvhNode(NativeSlice<Entity> entities, NativeList<BvhNode> nodes) : this()
		{
			var possiblePartitions = PartitionAxis.All;
			var entireBounds = new AxisAlignedBoundingBox(float.MaxValue, float.MinValue);
			foreach (var entity in entities)
				entireBounds = AxisAlignedBoundingBox.Enclose(entireBounds, entity.Bounds);

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
					Content = entities[0];
					Bounds = Content.Bounds;
					IsLeaf = true;
					break;

				default:
					var nodesPointer = (BvhNode*) nodes.GetUnsafePtr();

					var leftNode = new BvhNode(new NativeSlice<Entity>(entities, 0, n / 2), nodes);
					var rightNode = new BvhNode(new NativeSlice<Entity>(entities, n / 2), nodes);

					nodes.Add(leftNode);
					Left = nodesPointer + (nodes.Length - 1);

					nodes.Add(rightNode);
					Right = nodesPointer + (nodes.Length - 1);

					Bounds = AxisAlignedBoundingBox.Enclose(Left->Bounds, Right->Bounds);
					break;
			}
		}

		public IReadOnlyList<ValueTuple<AxisAlignedBoundingBox, int>> GetAllSubBounds(int depth = 0,
			List<ValueTuple<AxisAlignedBoundingBox, int>> workingList = null)
		{
			if (workingList == null)
				workingList = new List<ValueTuple<AxisAlignedBoundingBox, int>>();

			workingList.Add((Bounds, depth));

			if (!IsLeaf)
			{
				Left->GetAllSubBounds(depth + 1, workingList);
				Right->GetAllSubBounds(depth + 1, workingList);
			}

			return workingList;
		}
	}
}
#endif