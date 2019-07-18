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
		public bool Hit(Ray r, float tMin, float tMax)
		{
			// TODO: vectorize
			for (int a = 0; a < 3; a++)
			{
				float invDirection = 1 / r.Direction[a];
				float t0 = (Min[a] - r.Origin[a]) * invDirection;
				float t1 = (Max[a] - r.Origin[a]) * invDirection;

				if (invDirection < 0)
					Util.Swap(ref t0, ref t1);

				tMin = t0 > tMin ? t0 : tMin;
				tMax = t1 < tMax? t1 : tMax;

				if (tMax <= tMin)
					return false;
			}

			return true;
		}

		public static AxisAlignedBoundingBox Enclose(AxisAlignedBoundingBox lhs, AxisAlignedBoundingBox rhs) =>
			new AxisAlignedBoundingBox(min(lhs.Max, rhs.Min), max(lhs.Max, rhs.Max));
	}
}