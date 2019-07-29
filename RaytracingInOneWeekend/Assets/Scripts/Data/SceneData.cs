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
		[InlineButton(nameof(UpdateFromUnityCamera), "Update")]
		[SerializeField]
		Vector3 cameraPosition = default;
		[InlineButton(nameof(UpdateFromUnityCamera), "Update")]
		[SerializeField] Vector3 cameraTarget = default;
		[SerializeField] float cameraAperture = 0.1f;
		[SerializeField] [Range(0.001f, 180)] float cameraFieldOfView = 20;

		[Title("World")]
		[SerializeField] SphereData[] spheres = null;
		[SerializeField] [Range(1, 10000)] uint randomSeed = 1;
		[SerializeField] RandomSphereGroup[] randomSphereGroups = null;

		[Title("Sky")]
		[SerializeField] Color skyBottomColor = Color.white;
		[SerializeField] Color skyTopColor = new Color(0.5f, 0.7f, 1);

		public Vector3 CameraPosition => cameraPosition;
		public Vector3 CameraTarget => cameraTarget;
		public float CameraAperture => cameraAperture;
		public float CameraFieldOfView => cameraFieldOfView;
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

		public SceneData DeepClone()
		{
			SceneData clone = Instantiate(this);
			clone.hideFlags = HideFlags.HideAndDontSave;
			clone.name = $"{name} [RUNTIME COPY]";

			for (int i = 0; i < spheres.Length; i++)
			{
				SphereData sourceSphere = spheres[i];

				MaterialData clonedMaterial = null;
				if (sourceSphere.Material)
				{
					clonedMaterial = Instantiate(sourceSphere.Material);
					clonedMaterial.name = $"{sourceSphere.Material.name} [RUNTIME COPY]";
					clonedMaterial.hideFlags = HideFlags.HideAndDontSave;

					if (sourceSphere.Material.Albedo)
					{
						clonedMaterial.Albedo = Instantiate(sourceSphere.Material.Albedo);
						clonedMaterial.Albedo.name = $"{sourceSphere.Material.Albedo.name} [RUNTIME COPY]";
						clonedMaterial.Albedo.hideFlags = HideFlags.HideAndDontSave;
					}
				}

				var clonedSphere = new SphereData();
				UnityEditor.EditorUtility.CopySerializedManagedFieldsOnly(sourceSphere, clonedSphere);
				clonedSphere.Material = clonedMaterial;

				clone.spheres[i] = clonedSphere;
			}

			for (int i = 0; i < randomSphereGroups.Length; i++)
			{
				var sourceGroup = randomSphereGroups[i];

				var clonedGroup = new RandomSphereGroup();
				UnityEditor.EditorUtility.CopySerializedManagedFieldsOnly(sourceGroup, clonedGroup);

				clone.randomSphereGroups[i] = clonedGroup;
			}

			return clone;
		}
#endif

		public void UpdateFromUnityCamera()
		{
			var unityCamera = FindObjectOfType<UnityEngine.Camera>();
			cameraPosition = unityCamera.transform.position;
			cameraTarget = cameraPosition + unityCamera.transform.forward;
			cameraFieldOfView = unityCamera.fieldOfView;
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

		// TODO: this way of setting up the grid is pretty aggravating to work with
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
