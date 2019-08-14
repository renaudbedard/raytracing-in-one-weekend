using System;
using Unity.Mathematics;
using UnityEngine;

namespace RaytracerInOneWeekend
{
	[Serializable]
	public class BoxData
	{
		[SerializeField] Vector3 size = Vector3.one;

		public BoxData(float3 size)
		{
			this.size = size;
		}

		public Vector3 Size
		{
			get => size;
			set => size = value;
		}
	}
}