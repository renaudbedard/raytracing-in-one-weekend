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
#if BUFFERED_MATERIALS
		public readonly int MaterialIndex;
#else
		public readonly Material Material;
#endif

#if BUFFERED_MATERIALS
		public Sphere(float3 center, float radius, int materialIndex)
#else
		public Sphere(float3 center, float radius, Material material)
#endif
		{
			Center = center;
			Radius = radius;
			SquaredRadius = radius * radius;
#if BUFFERED_MATERIALS
			MaterialIndex = materialIndex;
#else
			Material = material;
#endif
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