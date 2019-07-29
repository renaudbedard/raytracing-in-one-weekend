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
		public readonly Entity Content;
		[NativeDisableUnsafePtrRestriction] public BvhNode* Left, Right;
		public readonly BvhNodeMetadata* Metadata;

		public bool IsLeaf => Content.Type != EntityType.None;

		public static int GetNodeCount(NativeArray<Entity> entities)
		{
			using (var tempNodes = new NativeList<BvhNode>(Allocator.Temp))
			using (var tempMetadata = new NativeList<BvhNodeMetadata>(Allocator.Temp))
			{
				tempNodes.Add(new BvhNode(entities, tempNodes, tempMetadata));
				return tempNodes.Length;
			}
		}

		public BvhNode(NativeSlice<Entity> entities, NativeList<BvhNode> nodes, NativeList<BvhNodeMetadata> metadata,
			int depth = 0, int rank = 0, int parentNodeId = 0) : this()
		{
			metadata.Add(default);
			Metadata = (BvhNodeMetadata*) metadata.GetUnsafePtr() + metadata.Length - 1;

			Metadata->Depth = depth;
			Metadata->Order = depth * 100 * nodes.Capacity + rank * 10 * nodes.Capacity + parentNodeId;

			var entireBounds = new AxisAlignedBoundingBox(float.MaxValue, float.MinValue);
			foreach (Entity entity in entities)
				entireBounds = AxisAlignedBoundingBox.Enclose(entireBounds, entity.Bounds);

			var biggestPartition = PartitionAxis.None;
			var biggestPartitionSize = float.MinValue;
			foreach (PartitionAxis partition in PartitionAxis.All.Enumerate())
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
					break;

				default:
					var leftNode = new BvhNode(new NativeSlice<Entity>(entities, 0, n / 2), nodes, metadata,
						depth + 1, 0, Metadata->Id);
					Metadata->LeftId = leftNode.Metadata->Id = nodes.Length;
					nodes.Add(leftNode);

					var rightNode = new BvhNode(new NativeSlice<Entity>(entities, n / 2), nodes, metadata,
						depth + 1, 1, Metadata->Id);
					Metadata->RightId = rightNode.Metadata->Id = nodes.Length;
					nodes.Add(rightNode);

					Bounds = AxisAlignedBoundingBox.Enclose(leftNode.Bounds, rightNode.Bounds);
					break;
			}
		}

		public void SetupPointers(NativeList<BvhNode> nodes)
		{
			if (IsLeaf) return;

			Left = Right = null;

			for (int i = 0; i < nodes.Length; i++)
			{
				if (nodes[i].Metadata->Id == Metadata->LeftId) Left = (BvhNode*) nodes.GetUnsafePtr() + i;
				if (nodes[i].Metadata->Id == Metadata->RightId) Right = (BvhNode*) nodes.GetUnsafePtr() + i;
				if (Left != null && Right != null) break;
			}

			Left->SetupPointers(nodes);
			Right->SetupPointers(nodes);
		}

		public IReadOnlyList<ValueTuple<AxisAlignedBoundingBox, int>> GetAllSubBounds(List<ValueTuple<AxisAlignedBoundingBox, int>> workingList = null)
		{
			if (workingList == null)
				workingList = new List<ValueTuple<AxisAlignedBoundingBox, int>>();

			workingList.Add((Bounds, Metadata->Depth));

			if (!IsLeaf)
			{
				Left->GetAllSubBounds(workingList);
				Right->GetAllSubBounds(workingList);
			}

			return workingList;
		}
	}

	struct BvhNodeMetadata
	{
		public int Id, LeftId, RightId, Depth, Order;
	}

	struct BvhNodeComparer : IComparer<BvhNode>
	{
		public unsafe int Compare(BvhNode lhs, BvhNode rhs)
		{
			return lhs.Metadata->Order - rhs.Metadata->Order;
		}
	}
}
#endif