using System;
using UnityEngine;

namespace RaytracerInOneWeekend
{
	[Serializable]
	class MeshData
	{
		[SerializeField] Mesh mesh;

		public Mesh Mesh
		{
			get => mesh;
			set => mesh = value;
		}
	}
}