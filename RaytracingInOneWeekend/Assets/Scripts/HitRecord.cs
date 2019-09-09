using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
	unsafe struct HitRecord
	{
		public readonly float Distance;
		public readonly float3 Point;
		public readonly float3 Normal;
		public readonly int EntityId;

		public HitRecord(float distance, float3 point, float3 normal, int entityId)
		{
			Distance = distance;
			Point = point;
			Normal = normal;
			EntityId = entityId;
		}
	}
}