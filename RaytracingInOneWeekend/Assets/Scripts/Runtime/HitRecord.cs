using System.Collections.Generic;
using Unity.Mathematics;

namespace Runtime
{
	unsafe struct HitRecord
	{
		public readonly float Distance;
		public readonly float3 Point;
		public readonly float3 Normal;
		public readonly float2 TexCoords;
		public Entity* EntityPtr;

		public HitRecord(float distance, float3 point, float3 normal, float2 texCoords) : this()
		{
			Distance = distance;
			Point = point;
			Normal = normal;
			TexCoords = texCoords;
		}

		public readonly struct DistanceComparer : IComparer<HitRecord>
		{
			public int Compare(HitRecord x, HitRecord y) => x.Distance.CompareTo(y.Distance);
		}
	}
}