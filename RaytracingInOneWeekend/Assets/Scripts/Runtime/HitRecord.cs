using System;
using Unity.Mathematics;

namespace Runtime
{
	unsafe struct HitRecord : IComparable<HitRecord>
	{
		public readonly float Distance;
		public readonly float3 Point;
		public readonly float3 Normal;
		public Entity* EntityPtr;
		public readonly bool InProbabilisticVolume;

		public HitRecord(float distance, float3 point, float3 normal, bool inProbabilisticVolume) : this()
		{
			Distance = distance;
			Point = point;
			Normal = normal;
			InProbabilisticVolume = inProbabilisticVolume;
		}

		public int CompareTo(HitRecord other)
		{
			return Distance.CompareTo(other.Distance);
		}
	}
}