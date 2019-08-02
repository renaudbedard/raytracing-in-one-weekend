using Unity.Mathematics;

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

		public float3 GetPoint(float t) => Origin + t * Direction;
	}
}