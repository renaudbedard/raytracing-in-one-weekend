#if !(SOA_SIMD || AOSOA_SIMD)
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	struct Sphere
	{
		public readonly float3 Center;
		public readonly float SquaredRadius;
		public readonly float Radius;
		public readonly Material Material;

		public Sphere(float3 center, float radius, Material material)
		{
			Center = center;
			Radius = radius;
			SquaredRadius = radius * radius;
			Material = material;
		}

		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				float absRadius = abs(Radius);
				return new AxisAlignedBoundingBox(Center - absRadius, Center + absRadius);
			}
		}
	}
}
#endif