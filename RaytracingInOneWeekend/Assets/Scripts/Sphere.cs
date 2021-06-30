using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	readonly struct Sphere
	{
		public readonly float SquaredRadius, Radius;

		public Sphere(float radius)
		{
			Radius = radius;
			SquaredRadius = radius * radius;
		}

		public float Pdf(float3 entityLocalRayOrigin)
		{
			float cosThetaMax = sqrt(1 - SquaredRadius / lengthsq(-entityLocalRayOrigin));
			float solidAngle = 2 * PI * (1 - cosThetaMax);
			return 1 / solidAngle;
		}

		// TODO: this could (should?) be view-dependent
		public float3 RandomPoint(ref RandomSource rng) => rng.NextFloat3Direction() * Radius;

		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				float absRadius = abs(Radius);
				return new AxisAlignedBoundingBox(-absRadius, absRadius);
			}
		}
	}
}