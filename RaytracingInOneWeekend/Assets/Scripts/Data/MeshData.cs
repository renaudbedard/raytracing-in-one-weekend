using System;
using UnityEngine;

namespace RaytracerInOneWeekend
{
	[Serializable]
	class MeshData
	{
		[SerializeField] MeshFilter meshFilter;

		public MeshFilter A
		{
			get => meshFilter;
			set => meshFilter = value;
		}
	}
}