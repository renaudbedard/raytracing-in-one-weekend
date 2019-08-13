using Unity.Mathematics;
using static Unity.Mathematics.math;
using UnityEngine;

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

		public AxisAlignedBoundingBox Bounds => new AxisAlignedBoundingBox(
			float3(From, -Mathf.Epsilon),
			float3(To, Mathf.Epsilon));
	}
}