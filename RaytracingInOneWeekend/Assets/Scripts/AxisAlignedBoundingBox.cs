using JetBrains.Annotations;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	struct AxisAlignedBoundingBox
	{
		public readonly float3 Min, Max;

		public AxisAlignedBoundingBox(float3 min, float3 max)
		{
			Min = min;
			Max = max;
		}

		[Pure]
		public bool Hit(float3 rayOrigin, float3 rayInvDirection, float tMin, float tMax)
		{
			// optimized algorithm from Roman Wiche
			// https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525

			float3 t0 = (Min - rayOrigin) * rayInvDirection;
			float3 t1 = (Max - rayOrigin) * rayInvDirection;

			tMin = max(tMin, cmax(min(t0, t1)));
			tMax = min(tMax, cmin(max(t0, t1)));

			return tMin < tMax;
		}

		public float3 Center => Min + (Max - Min) / 2;
		public float3 Size => Max - Min;

		public float3[] Corners => new []
		{
			float3(Min.x, Min.y, Min.z),
			float3(Min.x, Min.y, Max.z),
			float3(Min.x, Max.y, Min.z),
			float3(Max.x, Min.y, Min.z),
			float3(Min.x, Max.y, Max.z),
			float3(Max.x, Max.y, Min.z),
			float3(Max.x, Min.y, Max.z),
			float3(Max.x, Max.y, Max.z),
		};

		public static AxisAlignedBoundingBox Enclose(AxisAlignedBoundingBox lhs, AxisAlignedBoundingBox rhs)
		{
			return new AxisAlignedBoundingBox(min(lhs.Min, rhs.Min), max(lhs.Max, rhs.Max));
		}
	}
}