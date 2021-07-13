using System;
using UnityEngine;

namespace Unity
{
	[Serializable]
	class MeshData
	{
		[SerializeField] Mesh mesh;
		[SerializeField] bool faceNormals;

		public Mesh Mesh
		{
			get => mesh;
			set => mesh = value;
		}

		public bool FaceNormals
		{
			get => faceNormals;
			set => faceNormals = value;
		}
	}
}