using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
	struct HitRecord
	{
		public readonly float Distance;
		public readonly float3 Point;
		public readonly float3 Normal;
#if BUFFERED_MATERIALS || UNITY_SOA
		public readonly int MaterialIndex;
#else
		public readonly Material Material;
#endif

#if BUFFERED_MATERIALS || UNITY_SOA
		public HitRecord(float distance, float3 point, float3 normal, int materialIndex)
#else
		public HitRecord(float distance, float3 point, float3 normal, Material material)
#endif
		{
			Distance = distance;
			Point = point;
			Normal = normal;
#if BUFFERED_MATERIALS || UNITY_SOA
			MaterialIndex = materialIndex;
#else
			Material = material;
#endif
		}
	}
}