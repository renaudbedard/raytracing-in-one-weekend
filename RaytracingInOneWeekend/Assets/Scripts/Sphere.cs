#if !(SOA_SIMD || AOSOA_SIMD)
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	struct Sphere
	{
		public readonly float3 CenterFrom, CenterTo;
		public readonly float2 TimeRange;
		public readonly float SquaredRadius;
		public readonly float Radius;
		public readonly Material Material;

		public Sphere(float3 center, float radius, Material material)
		{
			CenterFrom = CenterTo = center;
			TimeRange = 0;
			Radius = radius;
			SquaredRadius = radius * radius;
			Material = material;
		}

		public Sphere(float3 centerFrom, float3 centerTo, float t0, float t1, float radius, Material material)
		{
			CenterFrom = centerFrom;
			CenterTo = centerTo;
			TimeRange = float2(t0, t1);
			Radius = radius;
			SquaredRadius = radius * radius;
			Material = material;
		}

		public float3 Center(float t) =>
			lerp(CenterFrom, CenterTo, saturate(unlerp(TimeRange[0], TimeRange[1], t)));

		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				float3 minCenter = min(CenterFrom, CenterTo);
				float3 maxCenter = max(CenterFrom, CenterTo);
				float absRadius = abs(Radius);
				return new AxisAlignedBoundingBox(minCenter - absRadius, maxCenter + absRadius);
			}
		}
	}
}
#endif