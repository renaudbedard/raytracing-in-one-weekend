using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	readonly struct Triangle
	{
		public readonly float3 A, B, C;

		// Precalculated data
		public readonly float3 AB, AC;
		public readonly float3 Normal;

		public Triangle(float3 a, float3 b, float3 c)
		{
			A = a;
			B = b;
			C = c;

			AB = B - A;
			AC = C - A;
			Normal = normalize(cross(AB, AC));
		}

		public AxisAlignedBoundingBox Bounds => new AxisAlignedBoundingBox(
			min(min(A, B), C) + abs(Normal) * -0.001f,
			max(max(A, B), C) + abs(Normal) * 0.001f);
	}
}