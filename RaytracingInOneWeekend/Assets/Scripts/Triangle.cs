using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	readonly struct Triangle
	{
		public readonly float3 A, B, C;

		public Triangle(float3 a, float3 b, float3 c)
		{
			A = a;
			B = b;
			C = c;
		}

		public AxisAlignedBoundingBox Bounds => new AxisAlignedBoundingBox(
			min(min(A, B), C),
			max(max(A, B), C));
	}
}