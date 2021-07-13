using System;
using System.Collections.Generic;
using System.Linq;
using Runtime;
using Sirenix.OdinInspector;
using UnityEngine;
#if ODIN_INSPECTOR

#else
using OdinMock;
#endif

namespace Unity
{
	[CreateAssetMenu]
	class SceneData : ScriptableObject
	{
		[Title("Camera")]
		[SerializeField]
		[LabelText("Position")]
		Vector3 cameraPosition = default;

		[SerializeField]
		[LabelText("Target Position")]
		Vector3 cameraTarget = default;

#if UNITY_EDITOR
		[ButtonGroup("Update Camera", -1)]
		public void UpdateFromGameView()
		{
			UnityEngine.Camera unityCamera = FindObjectOfType<UnityEngine.Camera>();

			cameraFieldOfView = unityCamera.fieldOfView;
			cameraPosition = unityCamera.transform.position;
			cameraTarget = cameraPosition + unityCamera.transform.forward;
		}

		[ButtonGroup("Update Camera", -1)]
		public void UpdateFromSceneView()
		{
			UnityEngine.Camera unityCamera = UnityEditor.SceneView.GetAllSceneCameras()[0];

			cameraPosition = unityCamera.transform.position;
			cameraTarget = cameraPosition + unityCamera.transform.forward;

			unityCamera = FindObjectOfType<UnityEngine.Camera>();
			unityCamera.transform.position = cameraPosition;
			unityCamera.transform.forward = cameraTarget - cameraPosition;
		}
#endif

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

		[SerializeField] RandomEntityGroup[] randomEntityGroups = null;

		[Title("Sky")]
		[SerializeField]
		SkyType skyType = SkyType.GradientSky;

		[SerializeField]
		[ColorUsage(false, true)]
		[LabelText("Bottom Color")]
		[ShowIf(nameof(skyType), SkyType.GradientSky)]
		Color skyBottomColor = Color.white;

		[SerializeField]
		[ColorUsage(false, true)]
		[LabelText("Top Color")]
		[ShowIf(nameof(skyType), SkyType.GradientSky)]
		Color skyTopColor = new Color(0.5f, 0.7f, 1);

		[SerializeField]
		[LabelText("Cubemap")]
		[ShowIf(nameof(skyType), SkyType.CubeMap)]
		UnityEngine.Cubemap skyCubemap = default;

		public Vector3 CameraPosition => cameraPosition;
		public Vector3 CameraTarget => cameraTarget;
		public float CameraAperture => cameraAperture;
		public float CameraFieldOfView => cameraFieldOfView;
		public uint RandomSeed => randomSeed;
		public IReadOnlyList<EntityData> Entities => entities;
		public IReadOnlyList<RandomEntityGroup> RandomEntityGroups => randomEntityGroups;

		public SkyType SkyType => skyType;
		public Color SkyBottomColor => skyBottomColor;
		public Color SkyTopColor => skyTopColor;
		public UnityEngine.Cubemap SkyCubemap => skyCubemap;

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

		public void MarkDirty() => dirty = true;

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
					clonedMaterial.Albedo = sourceEntity.Material.Albedo;
					clonedMaterial.Emission = sourceEntity.Material.Emission;
				}

				var clonedEntity = (EntityData) Activator.CreateInstance(sourceEntity.GetType());
				UnityEditor.EditorUtility.CopySerializedManagedFieldsOnly(sourceEntity, clonedEntity);
				clonedEntity.Material = clonedMaterial;

				clone.entities[i] = clonedEntity;
			}

			for (int i = 0; i < randomEntityGroups.Length; i++)
			{
				var sourceGroup = randomEntityGroups[i];

				var clonedGroup = new RandomEntityGroup();
				UnityEditor.EditorUtility.CopySerializedManagedFieldsOnly(sourceGroup, clonedGroup);

				clone.randomEntityGroups[i] = clonedGroup;
			}

			return clone;
		}

		public void AddEntity(EntityData entity)
		{
			Array.Resize(ref entities, entities.Length + 1);
			entities[entities.Length - 1] = entity;
			MarkDirty();
		}
#endif
	}

	enum RandomDistribution
	{
		DartThrowing,
		JitteredGrid
	}

	[Serializable]
	class RandomEntityGroup
	{
		public EntityType Type = EntityType.Sphere;
		public RandomDistribution Distribution = default;

		// TODO: this way of setting up the grid is pretty aggravating to work with
		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0.01f, 100)]
		public float PeriodX = 1;
		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0.01f, 100)]
		public float PeriodY = 1;
		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0.01f, 100f)]
		public float PeriodZ = 1;

		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[Range(0, 1)]
		public float PositionVariation = default;

		[ShowIf(nameof(Distribution), RandomDistribution.JitteredGrid)]
		[MinMaxSlider(0, 2, true)]
		public Vector2 ScaleVariationX = Vector2.one, ScaleVariationY = Vector2.one, ScaleVariationZ = Vector2.one;

		[ShowIf(nameof(Distribution), RandomDistribution.DartThrowing)]
		[Range(0, 1000)] public float TentativeCount = 1;

		public bool SkipOverlapTest = false;

		[Range(0, 2000)]
		public float SpreadX = default, SpreadY = default, SpreadZ = default;
		public Vector3 Offset = default;
		public bool OffsetByRadius = default;
		public Vector3 Rotation = default;

		[Range(0, 1)] public float MovementChance = 0;

		[HideIf(nameof(MovementChance), 0.0f)]
		[MinMaxSlider(-1, 1, true)]
		public Vector2 MovementXOffset = default, MovementYOffset = default, MovementZOffset = default;

		[MinMaxSlider(0, 100, true)] public Vector2 Radius = default;

		[HideIf(nameof(SkipOverlapTest))]
		[Range(0, 100)] public float MinDistance = default;

		[Range(0, 1)] public float LambertChance = 1;
		[Range(0, 1)] public float MetalChance = default;
		[Range(0, 1)] public float DieletricChance = default;
		[Range(0, 1)] public float LightChance = default;

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

		[HideIf(nameof(LightChance), 0.0f)]
		[GradientUsage(true)]
		public Gradient Emissive = default;
	}
}
