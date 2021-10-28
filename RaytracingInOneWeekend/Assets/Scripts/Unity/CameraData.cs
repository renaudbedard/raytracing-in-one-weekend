using UnityEngine;

namespace Unity
{
	[ExecuteAlways]
	public class CameraData : MonoBehaviour
	{
		[Range(0, 1)] public float ApertureSize;

		void Awake()
		{
			GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth;
		}
	}
}