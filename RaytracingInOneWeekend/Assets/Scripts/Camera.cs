using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
    struct Camera
    {
        public readonly float3 Origin;
        public readonly float3 LowerLeftCorner;
        public readonly float3 Horizontal;
        public readonly float3 Vertical;

        public Camera(float3 origin, float3 lowerLeftCorner, float3 horizontal, float3 vertical)
        {
            Origin = origin;
            LowerLeftCorner = lowerLeftCorner;
            Horizontal = horizontal;
            Vertical = vertical;
        }

        public Ray GetRay(float2 normalizedCoordinates) => new Ray(Origin,
            LowerLeftCorner + normalizedCoordinates.x * Horizontal + normalizedCoordinates.y * Vertical - Origin);
    }
}