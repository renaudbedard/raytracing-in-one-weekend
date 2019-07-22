#if BVH
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Assertions;

#if QUAD_BVH
using System.Linq;
using Unity.Mathematics;
using static Unity.Mathematics.math;
#endif

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

#if QUAD_BVH
	struct BvhNode
	{
		public readonly QuadAabb ElementBounds;
		public readonly Entity NorthEast, SouthEast, SouthWest, NorthWest;
		public readonly AxisAlignedBoundingBox Bounds;

		public BvhNode(NativeSlice<Entity> entities, NativeList<BvhNode> nodes, NativeList<QuadAabbData> aabbData)
		{
			int n = entities.Length;
			if (n <= 4)
			{
				NorthEast = entities[min(0, n - 1)];
				SouthEast = entities[min(1, n - 1)];
				SouthWest = entities[min(2, n - 1)];
				NorthWest = entities[min(3, n - 1)];
			}
			else
			{
				(NativeSlice<Entity>, NativeSlice<Entity>) Partition(NativeSlice<Entity> source)
				{
					var entireBounds = new AxisAlignedBoundingBox(float.MaxValue, float.MinValue);
					foreach (Entity entity in source)
						entireBounds = AxisAlignedBoundingBox.Enclose(entireBounds, entity.Bounds);

					PartitionAxis biggestPartition = PartitionAxis.All.Enumerate()
						.OrderByDescending(x => entireBounds.Size[x.GetAxisId()]).First();

					source.Sort(new EntityBoundsComparer(biggestPartition));

					return (new NativeSlice<Entity>(source, 0, source.Length / 2),
						new NativeSlice<Entity>(source, source.Length / 2));
				}

				unsafe Entity AddNode(NativeSlice<Entity> slice)
				{
					nodes.Add(new BvhNode(slice, nodes, aabbData));
					return new Entity((BvhNode*) nodes.GetUnsafePtr() + (nodes.Length - 1));
				}

				var (eastSlice, westSlice) = Partition(entities);
				var (northEastSlice, southEastSlice) = Partition(eastSlice);
				var (northWestSlice, southWestSlice) = Partition(westSlice);

				NorthEast = northEastSlice.Length == 1 ? northEastSlice[0] : AddNode(northEastSlice);
				SouthEast = southEastSlice.Length == 1 ? southEastSlice[0] : AddNode(southEastSlice);
				SouthWest = southWestSlice.Length == 1 ? southWestSlice[0] : AddNode(southWestSlice);
				NorthWest = northWestSlice.Length == 1 ? northWestSlice[0] : AddNode(northWestSlice);
			}

			aabbData.Add(default);
			unsafe
			{
				ElementBounds = new QuadAabb((QuadAabbData*) aabbData.GetUnsafePtr() + (aabbData.Length - 1),
					NorthEast.Bounds, SouthEast.Bounds, SouthWest.Bounds, NorthWest.Bounds);
			}
			Bounds = ElementBounds.Enclosure;
		}

		public IEnumerable<ValueTuple<AxisAlignedBoundingBox, int>> GetAllSubBounds(int depth = 0)
		{
			yield return (Bounds, depth);

			if (NorthEast.Type == EntityType.BvhNode)
				foreach ((var bounds, int subdepth) in NorthEast.AsNode.GetAllSubBounds(depth + 1))
					yield return (bounds, subdepth);
			else
				yield return (NorthEast.Bounds, depth + 1);

			if (SouthEast.Type == EntityType.BvhNode)
				foreach ((var bounds, int subdepth) in SouthEast.AsNode.GetAllSubBounds(depth + 1))
					yield return (bounds, subdepth);
			else
				yield return (SouthEast.Bounds, depth + 1);

			if (SouthWest.Type == EntityType.BvhNode)
				foreach ((var bounds, int subdepth) in SouthWest.AsNode.GetAllSubBounds(depth + 1))
					yield return (bounds, subdepth);
			else
				yield return (SouthWest.Bounds, depth + 1);

			if (NorthWest.Type == EntityType.BvhNode)
				foreach ((var bounds, int subdepth) in NorthWest.AsNode.GetAllSubBounds(depth + 1))
					yield return (bounds, subdepth);
			else
				yield return (NorthWest.Bounds, depth + 1);
		}
	}

#else
	struct BvhNode
	{
		public readonly AxisAlignedBoundingBox Bounds;
		public readonly Entity Left, Right;

		public unsafe BvhNode(NativeSlice<Entity> entities, NativeList<BvhNode> nodes)
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

			Bounds = AxisAlignedBoundingBox.Enclose(Left.Bounds, Right.Bounds);
		}

		public IEnumerable<ValueTuple<AxisAlignedBoundingBox, int>> GetAllSubBounds(int depth = 0)
		{
			yield return (Bounds, depth);

			if (Left.Type == EntityType.BvhNode)
				foreach ((var bounds, int subdepth) in Left.AsNode.GetAllSubBounds(depth + 1))
					yield return (bounds, subdepth);
			else
				yield return (Left.Bounds, depth + 1);

			if (Right.Type == EntityType.BvhNode)
				foreach ((var bounds, int subdepth) in Right.AsNode.GetAllSubBounds(depth + 1))
					yield return (bounds, subdepth);
			else
				yield return (Right.Bounds, depth + 1);
		}
	}
#endif
}
#endif