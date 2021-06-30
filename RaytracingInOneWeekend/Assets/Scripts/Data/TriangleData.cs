using System;
using Unity.Mathematics;
using UnityEngine;

namespace RaytracerInOneWeekend
{
	[Serializable]
	class TriangleData
	{
		[SerializeField] Vector3 a, b, c;

		public TriangleData(float3 a, float3 b, float3 c)
		{
			this.a = a;
			this.b = b;
			this.c = c;
		}

		public Vector3 A
		{
			get => a;
			set => a = value;
		}

		public Vector3 B
		{
			get => b;
			set => b = value;
		}

		public Vector3 C
		{
			get => c;
			set => c = value;
		}
	}
}