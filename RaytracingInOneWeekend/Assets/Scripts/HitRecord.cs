using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
    struct HitRecord
    {
        public readonly float Distance;
        public readonly float3 Point;
        public readonly float3 Normal;
        public readonly Material Material;

        public HitRecord(float distance, float3 point, float3 normal, Material material)
        {
            Distance = distance;
            Point = point;
            Normal = normal;
            Material = material;
        }
    }
}