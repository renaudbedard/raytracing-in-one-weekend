using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
	struct HitRecord
	{
		public readonly float Distance;
		public readonly float3 Point;
		public readonly float3 Normal;
		public readonly ushort MaterialIndex;

		public HitRecord(float distance, float3 point, float3 normal, ushort materialIndex)
		{
			Distance = distance;
			Point = point;
			Normal = normal;
			MaterialIndex = materialIndex;
		}
	}
}