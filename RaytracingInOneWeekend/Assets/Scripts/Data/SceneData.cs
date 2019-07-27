using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace RaytracerInOneWeekend
{
	[CreateAssetMenu]
	class SceneData : ScriptableObject
	{
		[Title("Camera")]
		[InlineButton(nameof(UpdateFromCameraTransform), "Update")]
		[SerializeField]
		Vector3 cameraPosition = default;
		[InlineButton(nameof(UpdateFromCameraTransform), "Update")]
		[SerializeField] Vector3 cameraTarget = default;
		[SerializeField] float cameraAperture = 0.1f;

		[Title("World")]
		[SerializeField] [Range(1, 10000)] uint randomSeed = 1;
		[SerializeField] SphereData[] spheres = null;
		[SerializeField] RandomSphereGroup[] randomSphereGroups = null;

		[Title("Sky")]
		[SerializeField] Color skyBottomColor = Color.white;
		[SerializeField] Color skyTopColor = new Color(0.5f, 0.7f, 1);

		public Vector3 CameraPosition => cameraPosition;
		public Vector3 CameraTarget => cameraTarget;
		public float CameraAperture => cameraAperture;
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

		void UpdateFromCameraTransform()
		{
			Transform cameraTransform = FindObjectOfType<UnityEngine.Camera>().transform;
			cameraPosition = cameraTransform.position;
			cameraTarget = cameraPosition + cameraTransform.forward;
		}
	}

	enum RandomDistribution
	{
		DartThrowing,
		JitteredGrid
	}

	[Serializable]
	class RandomSphereGroup
	{
		public RandomDistribution Distribution = default;

		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0.01f, 10)]
		public float PeriodX = 1;
		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0.01f, 10)]
		public float PeriodY = 1;
		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0.01f, 10f)]
		public float PeriodZ = 1;

		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0, 1)]
		public float Variation = default;

		[ShowIf(nameof(Distribution), RandomDistribution.DartThrowing)]
		[Range(0, 1000)] public float TentativeCount = 1;

		[MinMaxSlider(-50, 50, true)] public Vector2 CenterX = default;
		[MinMaxSlider(-50, 50, true)] public Vector2 CenterY = default;
		[MinMaxSlider(-50, 50, true)] public Vector2 CenterZ = default;

		[MinMaxSlider(0, 100, true)] public Vector2 Radius = default;

		[Range(0, 100)] public float MinDistance = default;

		[Range(0, 1)] public float LambertChance = 1;
		[Range(0, 1)] public float MetalChance = default;
		[Range(0, 1)] public float DieletricChance = default;

		[HideIf(nameof(LambertChance), 0.0f)]
		public Gradient DiffuseAlbedo = default;

		[HideIf(nameof(LambertChance), 0.0f)]
		[LabelWidth(175)]
		public bool DoubleSampleDiffuseAlbedo = default;

		[HideIf(nameof(MetalChance), 0.0f)]
		public Gradient MetalAlbedo = default;

		[HideIf(nameof(MetalChance), 0.0f)]
		[MinMaxSlider(0, 1, true)]
		public Vector2 Fuzz = default;

		[HideIf(nameof(DieletricChance), 0.0f)]
		[MinMaxSlider(0, 2.65f, true)]
		public Vector2 RefractiveIndex = default;
	}
}
