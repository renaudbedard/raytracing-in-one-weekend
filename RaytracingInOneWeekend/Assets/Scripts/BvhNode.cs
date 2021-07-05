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

		static readonly PartitionAxis[][] Permutations =
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
		public static int MaxDepth = 16;

		public readonly AxisAlignedBoundingBox Bounds;
		public readonly Entity* EntitiesStart;
		public readonly int EntityCount;
		[NativeDisableUnsafePtrRestriction] public BvhNode* Left, Right;
		public readonly BvhNodeMetadata* Metadata;

		public bool IsLeaf => EntitiesStart != null;

		public BvhNode(NativeSlice<Entity> entities, NativeList<BvhNode> nodes, NativeList<BvhNodeMetadata> metadata, NativeList<Entity> bvhEntities,
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

			if (depth == MaxDepth || entities.Length == 1)
			{
				EntitiesStart = (Entity*) bvhEntities.GetUnsafePtr() + bvhEntities.Length;
				// TODO: Sorting desc by entity size would make sense here
				bvhEntities.AddRangeNoResize(entities.GetUnsafePtr(), entities.Length);

				Bounds = entities[0].Bounds;
				for (int i = 1; i < entities.Length; i++)
					Bounds = AxisAlignedBoundingBox.Enclose(Bounds, entities[i].Bounds);

				EntityCount = entities.Length;
			}
			else
			{
				EntitiesStart = null;
				EntityCount = 0;

				int partitionLength = 0;
				float partitionStart = entities[0].Bounds.Min[biggestAxis];

				// decide the size of the partition according to the size of the entities
				foreach (Entity entity in entities)
				{
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

				var leftNode = new BvhNode(new NativeSlice<Entity>(entities, 0, partitionLength), nodes, metadata, bvhEntities,
					depth + 1, 0, Metadata->Id);
				Metadata->LeftId = leftNode.Metadata->Id = nodes.Length;
				nodes.AddNoResize(leftNode);

				var rightNode = new BvhNode(new NativeSlice<Entity>(entities, partitionLength), nodes, metadata, bvhEntities,
					depth + 1, 1, Metadata->Id);
				Metadata->RightId = rightNode.Metadata->Id = nodes.Length;
				nodes.AddNoResize(rightNode);

				Bounds = AxisAlignedBoundingBox.Enclose(leftNode.Bounds, rightNode.Bounds);
			}
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

	struct BvhNodeOrderComparer : IComparer<BvhNode>
	{
		public unsafe int Compare(BvhNode lhs, BvhNode rhs)
		{
			return lhs.Metadata->Order - rhs.Metadata->Order;
		}
	}

	readonly struct EntityBoundsComparer : IComparer<Entity>
	{
		readonly int axisId;

		public EntityBoundsComparer(PartitionAxis axis) => axisId = axis.GetAxisId();

		public int Compare(Entity lhs, Entity rhs)
		{
			return (int) sign(lhs.Bounds.Min[axisId] - rhs.Bounds.Min[axisId]);
		}
	}
}
#endif