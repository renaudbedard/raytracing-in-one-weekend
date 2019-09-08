using System;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	struct Sphere
	{
		public readonly float SquaredRadius, Radius;

		public Sphere(float radius)
		{
			Radius = radius;
			SquaredRadius = radius * radius;
		}

		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				float absRadius = abs(Radius);
				return new AxisAlignedBoundingBox(-absRadius, absRadius);
			}
		}
	}

#if BVH_SIMD
	struct Sphere4
	{
#pragma warning disable 649
		public float4 CenterX, CenterY, CenterZ, SquaredRadius;
#pragma warning restore 649
	}
#endif
}