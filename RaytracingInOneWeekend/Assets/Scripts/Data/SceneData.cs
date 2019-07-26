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
		[SerializeField] Vector3 cameraPosition = default;
#if ODIN_INSPECTOR
		[InlineButton(nameof(UpdateFromCameraTransform), "Update")]
#endif
		[SerializeField] Vector3 cameraTarget = default;

		[Title("World")]
		[SerializeField] uint randomSeed = 45573880;
		[SerializeField] SphereData[] spheres = null;

		[Title("Sky")]
		[SerializeField] Color skyBottomColor = Color.white;
		[SerializeField] Color skyTopColor = new Color(0.5f, 0.7f, 1);

		public Vector3 CameraPosition => cameraPosition;
		public Vector3 CameraTarget => cameraTarget;
		public uint RandomSeed => randomSeed;
		public IReadOnlyList<SphereData> Spheres => spheres;
		public Color SkyBottomColor => skyBottomColor;
		public Color SkyTopColor => skyTopColor;

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