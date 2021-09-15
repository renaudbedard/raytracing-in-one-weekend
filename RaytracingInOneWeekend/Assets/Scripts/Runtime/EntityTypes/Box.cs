using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime.EntityTypes
{
	readonly struct Box
	{
		public readonly float3 InverseExtents;
		public readonly float3 Extents;

		public Box(float3 size)
		{
			Extents = size / 2;
			InverseExtents = 1 / Extents;
		}

		public AxisAlignedBoundingBox Bounds => new AxisAlignedBoundingBox(float3(-Extents), float3(Extents));
	}
}