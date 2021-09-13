#if BVH
using System;
using System.Collections.Generic;
using Runtime;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine.Assertions;
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

	readonly unsafe struct BvhBuildingEntity
	{
		public readonly Entity* Entity;
		public readonly AxisAlignedBoundingBox Bounds;

		public BvhBuildingEntity(Entity* entity)
		{
			Entity = entity;

			switch (entity->Type)
			{
				case EntityType.Sphere: Bounds = ((Sphere*) entity->Content)->Bounds; break;
				case EntityType.Rect: Bounds = ((Rect*) entity->Content)->Bounds; break;
				case EntityType.Box: Bounds = ((Box*) entity->Content)->Bounds; break;
				case EntityType.Triangle: Bounds = ((Triangle*) entity->Content)->Bounds; break;
				default: throw new InvalidOperationException($"Unknown entity type : {entity->Type}");
			}

			float3* corners = stackalloc float3[8]
			{
				float3(Bounds.Min.x, Bounds.Min.y, Bounds.Min.z),
				float3(Bounds.Min.x, Bounds.Min.y, Bounds.Max.z),
				float3(Bounds.Min.x, Bounds.Max.y, Bounds.Min.z),
				float3(Bounds.Max.x, Bounds.Min.y, Bounds.Min.z),
				float3(Bounds.Min.x, Bounds.Max.y, Bounds.Max.z),
				float3(Bounds.Max.x, Bounds.Max.y, Bounds.Min.z),
				float3(Bounds.Max.x, Bounds.Min.y, Bounds.Max.z),
				float3(Bounds.Max.x, Bounds.Max.y, Bounds.Max.z)
			};
			var minimum = new float3(float.PositiveInfinity);
			var maximum = new float3(float.NegativeInfinity);

			if (entity->Moving)
			{
				float3 destinationPosition = entity->OriginTransform.pos + entity->DestinationOffset;
				var minTransform = new RigidTransform(entity->OriginTransform.rot, min(entity->OriginTransform.pos, destinationPosition));
				var maxTransform = new RigidTransform(entity->OriginTransform.rot, max(entity->OriginTransform.pos, destinationPosition));

				for (var i = 0; i < 8; i++)
				{
					float3 c = corners[i];
					minimum = min(minimum, transform(minTransform, c));
					maximum = max(maximum, transform(maxTransform, c));
				}
			}
			else
			{
				for (var i = 0; i < 8; i++)
				{
					float3 transformedCorner = transform(entity->OriginTransform, corners[i]);
					minimum = min(minimum, transformedCorner);
					maximum = max(maximum, transformedCorner);
				}
			}

			Bounds = new AxisAlignedBoundingBox(minimum, maximum);
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct CreateBvhBuildingEntitiesJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<Entity> Entities;
		[WriteOnly] public NativeArray<BvhBuildingEntity> BvhBuildingEntities;

		public unsafe void Execute(int index)
		{
			BvhBuildingEntities[index] = new BvhBuildingEntity((Entity*) Entities.GetUnsafeReadOnlyPtr() + index);
		}
	}

	unsafe class BvhNodeData
	{
		static readonly ProfilerMarker encloseEntireBoundsMarker = new ProfilerMarker("Enclose entire bounds");
		static readonly ProfilerMarker sortEntitiesMarker = new ProfilerMarker("Sort entities");
		static readonly ProfilerMarker determinePartitionSizeMarker = new ProfilerMarker("Determine partition size");

		public static int MaxDepth = 16;

		public readonly AxisAlignedBoundingBox Bounds;
		public readonly Entity* EntitiesStart;
		public readonly int EntityCount, Depth;
		public readonly BvhNodeData Left, Right;

		public bool IsLeaf => EntitiesStart != null;

		public BvhNodeData(NativeSlice<BvhBuildingEntity> entities, NativeList<Entity> bvhEntities, int depth = 0, PartitionAxis sortAxis = PartitionAxis.None)
		{
			Depth = depth;

			var entireBounds = new AxisAlignedBoundingBox(float.MaxValue, float.MinValue);
			using (encloseEntireBoundsMarker.Auto())
			{
				foreach (BvhBuildingEntity entity in entities)
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
				using (sortEntitiesMarker.Auto())
				{
					// Use a job for sorting, if we are above a certain threshold where it becomes faster to do so
					if (entities.Length > 64)
					{
						SortJob<BvhBuildingEntity, BvhBuildingEntityBoundsComparer> sortJob = entities.SortJob(new BvhBuildingEntityBoundsComparer(biggestPartition));
						JobHandle sortJobHandle = sortJob.Schedule();
						sortJobHandle.Complete();
					}
					else
						entities.Sort(new BvhBuildingEntityBoundsComparer(biggestPartition));
				}

			int biggestAxis = biggestPartition.GetAxisId();

			if (depth == MaxDepth || entities.Length == 1)
			{
				EntitiesStart = (Entity*) bvhEntities.GetUnsafePtr() + bvhEntities.Length;
				// TODO: Sorting desc by entity size would make sense here
				for (int i = 0; i < entities.Length; i++)
					bvhEntities.AddNoResize(*entities[i].Entity);

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
				using (determinePartitionSizeMarker.Auto())
				{
					foreach (BvhBuildingEntity entity in entities)
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

				Left = new BvhNodeData(new NativeSlice<BvhBuildingEntity>(entities, 0, partitionLength), bvhEntities, depth + 1, biggestPartition);
				Right = new BvhNodeData(new NativeSlice<BvhBuildingEntity>(entities, partitionLength), bvhEntities, depth + 1, biggestPartition);

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

	readonly struct BvhBuildingEntityBoundsComparer : IComparer<BvhBuildingEntity>
	{
		readonly int axisId;

		public BvhBuildingEntityBoundsComparer(PartitionAxis axis) => axisId = axis.GetAxisId();

		public int Compare(BvhBuildingEntity lhs, BvhBuildingEntity rhs)
		{
			return (int) sign(lhs.Bounds.Min[axisId] - rhs.Bounds.Min[axisId]);
		}
	}
}
#endif