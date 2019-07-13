using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
    struct HitRecord
    {
        public readonly float Distance;
        public readonly float3 Point;
        public readonly float3 Normal;

#if SOA_SPHERES
		public readonly ushort MaterialIndex;
#else
		public readonly Material Material;
#endif

#if SOA_SPHERES
		public HitRecord(float distance, float3 point, float3 normal, ushort materialIndex)
#else
        public HitRecord(float distance, float3 point, float3 normal, Material material)
#endif
		{
            Distance = distance;
            Point = point;
            Normal = normal;
#if SOA_SPHERES
			MaterialIndex = materialIndex;
#else
            Material = material;
#endif
        }
    }
}