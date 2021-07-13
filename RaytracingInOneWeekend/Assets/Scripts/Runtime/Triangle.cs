using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime
{
	readonly struct Triangle
	{
		public readonly float3x3 Data; // { 0: v2 - v0, 1: v1 - v0, 2: v0 }
		public readonly float3x3 Normals;

		// Face Normal
		public Triangle(float3 v1, float3 v2, float3 v3)
		{
			Data = float3x3(v3 - v1, v2 - v1, v1);
			float3 faceNormal = normalize(cross(Data[1], Data[0]));
			Normals = float3x3(faceNormal, faceNormal, faceNormal);
		}

		// Vertex Normals
		public Triangle(float3 v1, float3 v2, float3 v3, float3 n1, float3 n2, float3 n3)
		{
			Data = float3x3(v3 - v1, v2 - v1, v1);
			Normals = float3x3(normalize(n1), normalize(n2), normalize(n3));
		}

		public AxisAlignedBoundingBox Bounds
		{
			get
			{
				var vertices = float3x3(Data[2], Data[1] + Data[2], Data[0] + Data[2]);
				var absNormals = float3x3(abs(Normals[0]), abs(Normals[1]), abs(Normals[2]));
				var negativeOffset = vertices - absNormals * 0.001f;
				var positiveOffset = vertices + absNormals * 0.001f;
				return new AxisAlignedBoundingBox(
					min(min(negativeOffset[0], negativeOffset[1]), negativeOffset[2]),
					max(max(positiveOffset[0], positiveOffset[1]), positiveOffset[2]));
			}
		}
	}
}