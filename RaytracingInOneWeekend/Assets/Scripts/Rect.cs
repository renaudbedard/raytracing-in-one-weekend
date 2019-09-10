using Unity.Burst;
using Unity.Mathematics;
using UnityEngine.Assertions;
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

		[BurstDiscard]
		void Validate(Ray r, HitRecord rec)
		{
			Assert.IsTrue(length(r.Direction).AlmostEquals(1),
				$"Ray direction was assumed to be unit-length; length was {length(r.Direction):0.#######}");
			Assert.IsTrue(length(rec.Normal).AlmostEquals(1),
				$"HitRecord normal was assumed to be unit-length; length was {length(rec.Normal):0.#######}");
			Assert.IsTrue( dot(r.Direction, -rec.Normal) > 0,
				$"Cosine was assumed to be greater than 0; it was {dot(r.Direction, -rec.Normal):0.#######}");
		}

		public float PdfValue(Ray r, HitRecord rec)
		{
			Validate(r, rec);
			float area = Area;
			float distanceSquared = rec.Distance * rec.Distance;
			float cosine = dot(r.Direction, -rec.Normal);
			return distanceSquared / (cosine * area);
		}

		public float3 RandomPoint(ref Random rng) => float3(rng.NextFloat2(From, To), 0);

		public AxisAlignedBoundingBox Bounds => new AxisAlignedBoundingBox(
			float3(From, -0.001f),
			float3(To, 0.001f));
	}
}