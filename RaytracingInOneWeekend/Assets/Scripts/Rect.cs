using Unity.Mathematics;
using static Unity.Mathematics.math;
using UnityEngine;

namespace RaytracerInOneWeekend
{
	// An axis-aligned rectangle in the XY plane
	struct Rect
	{
		public readonly float Distance;
		public readonly float2 From, To;
		public readonly Material Material;

		public Rect(float distance, float2 center, float2 size, Material material)
		{
			Distance = distance;
			From = center - size / 2;
			To = center + size / 2;
			Material = material;
		}

		public AxisAlignedBoundingBox Bounds => new AxisAlignedBoundingBox(
			float3(From, Distance - Mathf.Epsilon),
			float3(To, Distance + Mathf.Epsilon));
	}
}