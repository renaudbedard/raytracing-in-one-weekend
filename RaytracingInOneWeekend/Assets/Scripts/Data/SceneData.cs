using System;
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
		[SerializeField]
		Vector3 cameraPosition = default;
#if ODIN_INSPECTOR
		[InlineButton(nameof(UpdateFromCameraTransform), "Update")]
#endif
		[SerializeField] Vector3 cameraTarget = default;

		[Title("World")] [SerializeField] uint randomSeed = 45573880;
		[SerializeField] SphereData[] spheres = null;
		[SerializeField] RandomSphereGroup[] randomSphereGroups = null;

		[Title("Sky")] [SerializeField] Color skyBottomColor = Color.white;
		[SerializeField] Color skyTopColor = new Color(0.5f, 0.7f, 1);

		public Vector3 CameraPosition => cameraPosition;
		public Vector3 CameraTarget => cameraTarget;
		public uint RandomSeed => randomSeed;
		public IReadOnlyList<SphereData> Spheres => spheres;
		public IReadOnlyList<RandomSphereGroup> RandomSphereGroups => randomSphereGroups;
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

	enum RandomDistribution
	{
		WhiteNoise,
		JitteredGrid
	}

	[Serializable]
	class RandomSphereGroup
	{
		public RandomDistribution Distribution = default;

		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(1, 100)]
		public int PeriodX = default;
		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(1, 100)]
		public int PeriodY = default;
		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(1, 100)]
		public int PeriodZ = default;

		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0, 1)]
		public float Variation = default;

		[ShowIf(nameof(Distribution), RandomDistribution.WhiteNoise)]
		[Range(0, 1000)] public float Count = 1;

		[MinMaxSlider(-50, 50, true)] public Vector2 CenterX = default;
		[MinMaxSlider(-50, 50, true)] public Vector2 CenterY = default;
		[MinMaxSlider(-50, 50, true)] public Vector2 CenterZ = default;

		[MinMaxSlider(-50, 50, true)] public Vector2 AvoidanceX = default;
		[MinMaxSlider(-50, 50, true)] public Vector2 AvoidanceY = default;
		[MinMaxSlider(-50, 50, true)] public Vector2 AvoidanceZ = default;

		[MinMaxSlider(0, 10, true)] public Vector2 Radius = default;

		[Range(0, 1)] public float LambertianProbability = 1;
		[Range(0, 1)] public float MetalProbability = default;
		[Range(0, 1)] public float DieletricProbability = default;

		public Gradient AlbedoColor = default;

		[MinMaxSlider(0, 1, true)] public Vector2 Fuzz = default;
		[MinMaxSlider(0, 2.65f, true)] public Vector2 RefractiveIndex = default;
	}
}