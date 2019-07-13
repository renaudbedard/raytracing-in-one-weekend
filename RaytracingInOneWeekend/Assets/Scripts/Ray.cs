using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
	struct Ray
	{
		public readonly float3 Origin;
		public readonly float3 Direction;

		public Ray(float3 origin, float3 direction)
		{
			Origin = origin;
			Direction = direction;
		}

		public float3 GetPoint(float t) => Origin + t * Direction;
	}
}