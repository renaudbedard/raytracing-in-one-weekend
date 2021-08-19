using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime
{
	readonly struct AxisAlignedBoundingBox
	{
		public readonly float3 Min, Max;

		public AxisAlignedBoundingBox(float3 min, float3 max)
		{
			Min = min;
			Max = max;
		}

		public float3 Center => Min + (Max - Min) / 2;
		public float3 Size => Max - Min;

		public float3[] Corners => new []
		{
			float3(Min.x, Min.y, Min.z),
			float3(Min.x, Min.y, Max.z),
			float3(Min.x, Max.y, Min.z),
			float3(Max.x, Min.y, Min.z),
			float3(Min.x, Max.y, Max.z),
			float3(Max.x, Max.y, Min.z),
			float3(Max.x, Min.y, Max.z),
			float3(Max.x, Max.y, Max.z),
		};

		public static AxisAlignedBoundingBox Enclose(AxisAlignedBoundingBox lhs, AxisAlignedBoundingBox rhs)
		{
			return new AxisAlignedBoundingBox(min(lhs.Min, rhs.Min), max(lhs.Max, rhs.Max));
		}
	}
}