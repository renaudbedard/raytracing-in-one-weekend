using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	struct Ray
	{
		public readonly float3 Origin;
		public readonly float3 Direction;
		public readonly float Time;

		public Ray(float3 origin, float3 direction, float time = 0)
		{
			Origin = origin;
			Direction = direction;
			Time = time;
		}

		public Ray OffsetTowards(float3 normal)
		{
			// from Listing 6-1 (page 84) in Ray Tracing Gems
			// http://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.5.pdf

			const float epsilon = 1.0f / 32.0f;
			const float floatScale = 1.0f / 65536.0f;
			const float intScale = 256.0f;

			int3 intOffset = int3(intScale * normal);
			float3 intPoint = asfloat(asint(Origin) + select(intOffset, -intOffset, Origin < 0));

			return new Ray(select(intPoint, Origin + floatScale * normal, abs(Origin) < epsilon),
				Direction, Time);

			// ----

			// el naive implementationne
			// return new Ray(Origin + 0.001f * normal, Direction, Time);
		}

		public float3 GetPoint(float t) => Origin + t * Direction;
	}
}