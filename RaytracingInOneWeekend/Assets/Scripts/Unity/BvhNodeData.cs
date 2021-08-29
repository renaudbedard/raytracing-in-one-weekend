#if BVH
using System;
using System.Collections.Generic;
using Runtime;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Assertions;
using Util;
using static Unity.Mathematics.math;

namespace Unity
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

	unsafe class BvhNodeData
	{
		public static int MaxDepth = 16;

		public readonly AxisAlignedBoundingBox Bounds;
		public readonly Entity* EntitiesStart;
		public readonly int EntityCount, Depth;
		public readonly BvhNodeData Left, Right;

		public bool IsLeaf => EntitiesStart != null;

		public BvhNodeData(NativeSlice<Entity> entities, NativeList<Entity> bvhEntities, int depth = 0, PartitionAxis sortAxis = PartitionAxis.None)
		{
			Depth = depth;

			var entireBounds = new AxisAlignedBoundingBox(float.MaxValue, float.MinValue);
			using (new CumulativeStopwatch("Enclose entire bounds"))
			{
				foreach (Entity entity in entities)
					entireBounds = AxisAlignedBoundingBox.Enclose(entireBounds, entity.Bounds);
			}

			var biggestPartition = PartitionAxis.None;
			var biggestPartitionSize = float.MinValue;
			float3 entireSize = entireBounds.Size;
			foreach (PartitionAxis partition in PartitionAxis.All.Enumerate())
			{
				float size = entireSize[partition.GetAxisId()];
				if (size > biggestPartitionSize)
				{
					biggestPartition = partition;
					biggestPartitionSize = size;
				}
			}

			if (sortAxis != biggestPartition)
				using (new CumulativeStopwatch("Sort entities"))
				{
					// Use a job for sorting, if we are above a certain threshold where it becomes faster to do so
					if (entities.Length > 64)
					{
						SortJob<Entity, EntityBoundsComparer> sortJob = entities.SortJob(new EntityBoundsComparer(biggestPartition));
						JobHandle sortJobHandle = sortJob.Schedule();
						sortJobHandle.Complete();
					}
					else
						entities.Sort(new EntityBoundsComparer(biggestPartition));
				}

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
				using (new CumulativeStopwatch("Determine partition size"))
				{
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
				}

				// ensure we have at least 1 entity in each partition
				if (partitionLength == entities.Length)
					partitionLength--;

				Left = new BvhNodeData(new NativeSlice<Entity>(entities, 0, partitionLength), bvhEntities, depth + 1, biggestPartition);
				Right = new BvhNodeData(new NativeSlice<Entity>(entities, partitionLength), bvhEntities, depth + 1, biggestPartition);

				Bounds = AxisAlignedBoundingBox.Enclose(Left.Bounds, Right.Bounds);
			}
		}

		public int ChildCount
		{
			get
			{
				if (IsLeaf)
					return 1;

				return Left.ChildCount + Right.ChildCount + 1;
			}
		}

		public IReadOnlyList<ValueTuple<AxisAlignedBoundingBox, int>> GetAllSubBounds(List<ValueTuple<AxisAlignedBoundingBox, int>> workingList = null)
		{
			workingList ??= new List<ValueTuple<AxisAlignedBoundingBox, int>>();
			workingList.Add((Bounds, Depth));

			if (!IsLeaf)
			{
				Left.GetAllSubBounds(workingList);
				Right.GetAllSubBounds(workingList);
			}

			return workingList;
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