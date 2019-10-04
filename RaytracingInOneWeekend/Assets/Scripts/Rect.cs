using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	// An axis-aligned rectangle in the XY plane
	struct Rect
	{
		public readonly float2 From, To;

		public Rect(float2 size)
		{
			From = -size / 2;
			To = size / 2;
		}

		public float Area
		{
			get
			{
				float2 size = To - From;
				return size.x * size.y;
			}
		}

		public float Pdf(float3 rayDirection, float hitDistance, float3 hitNormal)
		{
			float distanceSquared = hitDistance * hitDistance;
			float cosine = dot(rayDirection, -hitNormal);
			return distanceSquared / (cosine * Area);
		}

		public float3 RandomPoint(ref Random rng) => float3(rng.NextFloat2(From, To), 0);

		public AxisAlignedBoundingBox Bounds => new AxisAlignedBoundingBox(
			float3(From, -0.001f),
			float3(To, 0.001f));
	}
}