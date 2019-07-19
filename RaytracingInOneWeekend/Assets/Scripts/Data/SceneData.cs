using System.Collections.Generic;
using System.Linq;
using UnityEngine;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using Title = UnityEngine.HeaderAttribute;
#endif

namespace RaytracerInOneWeekend
{
	[CreateAssetMenu]
	class SceneData : ScriptableObject
	{
		[Title("Camera")]
#if ODIN_INSPECTOR
		[InlineButton(nameof(UpdateFromCameraTransform), "Update")]
#endif
		[SerializeField] Vector3 cameraPosition;
#if ODIN_INSPECTOR
		[InlineButton(nameof(UpdateFromCameraTransform), "Update")]
#endif
		[SerializeField] Vector3 cameraTarget;

		[Title("World")]
		[SerializeField] bool randomScene = true;

#if ODIN_INSPECTOR
		[ShowIf(nameof(randomScene))]
#endif
		[SerializeField] uint seed = 45573880;

#if ODIN_INSPECTOR
		[HideIf(nameof(randomScene))]
#endif
		[SerializeField] SphereData[] spheres = null;

		public Vector3 CameraPosition => cameraPosition;
		public Vector3 CameraTarget => cameraTarget;
		public bool RandomScene => randomScene;
		public uint Seed => seed;
		public IReadOnlyList<SphereData> Spheres => spheres;

#if UNITY_EDITOR
		bool dirty;
		public bool Dirty => dirty || Spheres.Any(x => x.Dirty);

		public void ClearDirty()
		{
			dirty = false;
			foreach (SphereData sphere in Spheres)
				sphere.ClearDirty();
		}

		void OnValidate()
		{
			dirty = true;
		}
#endif

#if ODIN_INSPECTOR
		void UpdateFromCameraTransform()
		{
			Transform cameraTransform = FindObjectOfType<UnityEngine.Camera>().transform;
			cameraPosition = cameraTransform.position;
			cameraTarget = cameraPosition + cameraTransform.forward;
		}
#endif
	}
}