using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime.EntityTypes
{
	readonly struct Sphere
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
}