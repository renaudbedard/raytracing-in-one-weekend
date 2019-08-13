using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	struct Box
	{
		public readonly float3 Size;

		public Box(float3 size)
		{
			Size = size;
		}

		public AxisAlignedBoundingBox Bounds => new AxisAlignedBoundingBox(float3(-Size / 2), float3(Size / 2));
	}
}