using UnityEngine;
namespace Unity
{
	public class MeshData : MonoBehaviour
	{
		[SerializeField] bool faceNormals;

		public bool FaceNormals => faceNormals;
	}
}