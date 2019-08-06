using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;
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
		[InlineButton(nameof(UpdateFromUnityCamera), "      Update      ")]
		[SerializeField]
		[LabelText("Position")]
		Vector3 cameraPosition = default;

		[InlineButton(nameof(UpdateFromUnityCamera), "      Update      ")]
		[SerializeField]
		[LabelText("Target Position")]
		Vector3 cameraTarget = default;

		[SerializeField]
		[LabelText("Aperture Size")]
		float cameraAperture = 0.1f;

		[SerializeField]
		[LabelText("Field Of View (°)")]
		[Range(0.001f, 180)]
		float cameraFieldOfView = 20;

		[Title("World")]
		[SerializeField] EntityData[] entities = null;

		[SerializeField]
		[Range(1, 10000)]
		uint randomSeed = 1;

		[SerializeField] RandomSphereGroup[] randomSphereGroups = null;

		[Title("Sky")]
		[SerializeField]
		[ColorUsage(false, true)]
		[LabelText("Bottom Color")]
		Color skyBottomColor = Color.white;

		[SerializeField]
		[ColorUsage(false, true)]
		[LabelText("Top Color")]
		Color skyTopColor = new Color(0.5f, 0.7f, 1);

		public Vector3 CameraPosition => cameraPosition;
		public Vector3 CameraTarget => cameraTarget;
		public float CameraAperture => cameraAperture;
		public float CameraFieldOfView => cameraFieldOfView;
		public uint RandomSeed => randomSeed;
		public IReadOnlyList<EntityData> Entities => entities;
		public IReadOnlyList<RandomSphereGroup> RandomSphereGroups => randomSphereGroups;
		public Color SkyBottomColor => skyBottomColor;
		public Color SkyTopColor => skyTopColor;

#if UNITY_EDITOR
		bool dirty;
		public bool Dirty => dirty || (entities != null && entities.Any(x => x.Dirty));

		public void ClearDirty()
		{
			dirty = false;

			if (entities != null)
				foreach (EntityData entity in entities)
					entity.ClearDirty();
		}

		void OnValidate()
		{
			dirty = true;
		}

		public SceneData DeepClone()
		{
			SceneData clone = Instantiate(this);
			clone.hideFlags = HideFlags.HideAndDontSave;
			clone.name = $"{name} [COPY]";

			for (int i = 0; i < entities.Length; i++)
			{
				EntityData sourceEntity = entities[i];

				MaterialData clonedMaterial = null;
				if (sourceEntity.Material)
				{
					clonedMaterial = Instantiate(sourceEntity.Material);
					clonedMaterial.name = $"{sourceEntity.Material.name} [COPY]";
					clonedMaterial.hideFlags = HideFlags.HideAndDontSave;

					if (sourceEntity.Material.Albedo)
					{
						clonedMaterial.Albedo = Instantiate(sourceEntity.Material.Albedo);
						clonedMaterial.Albedo.name = $"{sourceEntity.Material.Albedo.name} [COPY]";
						clonedMaterial.Albedo.hideFlags = HideFlags.HideAndDontSave;
					}
				}

				var clonedEntity = (EntityData) Activator.CreateInstance(sourceEntity.GetType());
				UnityEditor.EditorUtility.CopySerializedManagedFieldsOnly(sourceEntity, clonedEntity);
				clonedEntity.Material = clonedMaterial;

				clone.entities[i] = clonedEntity;
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

		[MinMaxSlider(-50, 50, true)]
		public Vector2 CenterX = default, CenterY = default, CenterZ = default;

		[Range(0, 1)] public float MovementChance = 0;

		[HideIf(nameof(MovementChance), 0.0f)]
		[MinMaxSlider(-1, 1, true)]
		public Vector2 MovementXOffset = default, MovementYOffset = default, MovementZOffset = default;

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
