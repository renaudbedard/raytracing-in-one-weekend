#if BVH
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Assertions;
using static Unity.Mathematics.math;

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

		private static readonly PartitionAxis[][] Permutations =
		{
			new PartitionAxis[0],
			new [] { PartitionAxis.X },
			new [] { PartitionAxis.Y },
			new [] { PartitionAxis.X, PartitionAxis.Y },
			new [] { PartitionAxis.Z },
			new [] { PartitionAxis.X, PartitionAxis.Z },
			new [] { PartitionAxis.Y, PartitionAxis.Z },
			new [] { PartitionAxis.X, PartitionAxis.Y, PartitionAxis.Z },
		};

		public static PartitionAxis[] Enumerate(this PartitionAxis axis) => Permutations[(int) axis];
	}

	unsafe struct BvhNode
	{
		public readonly AxisAlignedBoundingBox Bounds;
		public readonly int EntityId;
		[NativeDisableUnsafePtrRestriction] public BvhNode* Left, Right;
		public readonly BvhNodeMetadata* Metadata;

		public bool IsLeaf => EntityId != -1;

		public BvhNode(NativeSlice<Entity> entities, NativeList<BvhNode> nodes, NativeList<BvhNodeMetadata> metadata,
			int depth = 0, int rank = 0, int parentNodeId = 0) : this()
		{
			metadata.AddNoResize(default);
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
			int biggestAxis = biggestPartition.GetAxisId();

			switch (entities.Length)
			{
				case 1:
					EntityId = entities[0].Id;
					Bounds = entities[0].Bounds;
					break;

				default:
					EntityId = -1;

					int partitionLength = 0;
					float partitionStart = entities[0].Bounds.Min[biggestAxis];

					// decide the size of the partition according to the size of the entities
					for (var i = 0; i < entities.Length; i++)
					{
						Entity entity = entities[i];
						partitionLength++;
						AxisAlignedBoundingBox bounds = entity.Bounds;
						if (bounds.Min[biggestAxis] - partitionStart > biggestPartitionSize / 2 ||
							bounds.Size[biggestAxis] > biggestPartitionSize / 2)
						{
							break;
						}
					}

					// ensure we have at least 1 entity in each partition
					if (partitionLength == entities.Length)
						partitionLength--;

					var leftNode = new BvhNode(new NativeSlice<Entity>(entities, 0, partitionLength), nodes, metadata,
						depth + 1, 0, Metadata->Id);
					Metadata->LeftId = leftNode.Metadata->Id = nodes.Length;
					nodes.AddNoResize(leftNode);

					var rightNode = new BvhNode(new NativeSlice<Entity>(entities, partitionLength), nodes, metadata,
						depth + 1, 1, Metadata->Id);
					Metadata->RightId = rightNode.Metadata->Id = nodes.Length;
					nodes.AddNoResize(rightNode);

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

			Assert.IsFalse(Left == null || Right == null, "Subnodes should not be null");

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

	struct EntityBoundsComparer : IComparer<Entity>
	{
		readonly PartitionAxis axis;

		public EntityBoundsComparer(PartitionAxis axis) => this.axis = axis;

		public int Compare(Entity lhs, Entity rhs)
		{
			int axisId = axis.GetAxisId();
			return (int) sign(lhs.Bounds.Min[axisId] - rhs.Bounds.Min[axisId]);
		}
	}
}
#endif