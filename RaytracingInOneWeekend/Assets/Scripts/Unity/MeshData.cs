using System;
using UnityEngine;

namespace Unity
{
	[Serializable]
	class MeshData
	{
		[SerializeField] Mesh mesh;
		[SerializeField] bool faceNormals;
		[SerializeField] float scale = 1;

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

		public float Scale
		{
			get => scale;
			set => scale = value;
		}
	}
}