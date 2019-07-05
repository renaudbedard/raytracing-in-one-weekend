using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
    struct Camera
    {
        public readonly float3 Origin;
        public readonly float3 LowerLeftCorner;
        public readonly float3 Horizontal;
        public readonly float3 Vertical;

        public Camera(float3 origin, float3 lookAt, float3 up, float verticalFov, float aspect)
        {
            float theta = verticalFov * PI / 180;
            float halfHeight = tan(theta / 2);
            float halfWidth = aspect * halfHeight;

            float3 forward = normalize(origin - lookAt);
            float3 right = normalize(cross(up, forward));
            up = cross(forward, right);

            LowerLeftCorner = origin - halfWidth * right - halfHeight * up - forward;
            Horizontal = 2 * halfWidth * right;
            Vertical = 2 * halfHeight * up;
            Origin = origin;
        }

        public Ray GetRay(float2 normalizedCoordinates) => new Ray(Origin,
            LowerLeftCorner + normalizedCoordinates.x * Horizontal + normalizedCoordinates.y * Vertical - Origin);
    }
}