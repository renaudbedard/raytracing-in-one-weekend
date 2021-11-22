using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime
{
	readonly struct View
	{
		public readonly float3 Origin;
		public readonly float3 LowerLeftCorner;
		public readonly float3 Horizontal, Vertical;

		public readonly float3 Forward, Up, Right;

		public readonly float LensRadius;

		public View(float3 origin, float3 lookAt, float3 up, float verticalFov, float aspect, float aperture, float focusDistance)
		{
			LensRadius = aperture / 2;

			float theta = verticalFov * PI / 180;
			float halfHeight = tan(theta / 2);
			float halfWidth = aspect * halfHeight;

			Forward = normalize(origin - lookAt);
			Right = normalize(cross(Forward, up));
			Up = cross(Right, Forward);

			LowerLeftCorner = halfWidth * focusDistance * -Right +
							  halfHeight * focusDistance * -Up +
							  focusDistance * -Forward;

			Horizontal = 2 * halfWidth * focusDistance * Right;
			Vertical = 2 * halfHeight * focusDistance * Up;

			Origin = origin;
		}

		public Ray GetRay(float2 normalizedCoordinates, ref RandomSource rng)
		{
			float2 rd = LensRadius == 0 ? 0 : LensRadius * rng.InUnitDisk();
			float3 offset = Right * rd.x + Up * rd.y;

			return new Ray(Origin + offset,
				normalize(LowerLeftCorner - offset +
				          normalizedCoordinates.x * Horizontal +
				          normalizedCoordinates.y * Vertical),
				rng.NextFloat());
		}
	}
}