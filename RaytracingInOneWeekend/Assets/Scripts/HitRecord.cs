using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
	struct HitRecord
	{
		public readonly float Distance;
		public readonly float3 Point;
		public readonly float3 Normal;
#if BUFFERED_MATERIALS
		public readonly ushort MaterialIndex;
#else
		public readonly Material Material;
#endif

		public HitRecord(float distance, float3 point, float3 normal,
#if BUFFERED_MATERIALS			
			ushort materialIndex)
#else
			Material material)
#endif
		{
			Distance = distance;
			Point = point;
			Normal = normal;
#if BUFFERED_MATERIALS
			MaterialIndex = materialIndex;
#else
			Material = material;
#endif
		}
	}
}