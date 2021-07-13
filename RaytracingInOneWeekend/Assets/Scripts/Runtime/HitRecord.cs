using Unity.Mathematics;

namespace Runtime
{
	unsafe struct HitRecord
	{
		public readonly float Distance;
		public readonly float3 Point;
		public readonly float3 Normal;
		public Entity* EntityPtr;

		public HitRecord(float distance, float3 point, float3 normal) : this()
		{
			Distance = distance;
			Point = point;
			Normal = normal;
		}
	}
}