using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;
#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace RaytracerInOneWeekend
{
	[CreateAssetMenu]
	class MaterialData : ScriptableObject
	{
		[SerializeField] MaterialType type = MaterialType.None;

		[ShowIf(nameof(TextureCanScale))]
		[SerializeField] Vector2 textureScale = Vector2.one;

		[FormerlySerializedAs("fuzz")]
		[ShowIf(nameof(Type), MaterialType.Metal)]
		[Range(0, 1)] [SerializeField] float roughness = 0;

		[ShowIf(nameof(Type), MaterialType.Dielectric)]
		[Range(1, 2.65f)] [SerializeField] float refractiveIndex = 1;

		[ShowIf(nameof(Type), MaterialType.ProbabilisticVolume)]
		[Range(0, 1)] [SerializeField] float density = 1;

#if UNITY_EDITOR
		[AssetList]
		[ShowIf(nameof(AlbedoSupported))]
#endif
		[SerializeField] TextureData albedo = null;
#if UNITY_EDITOR
		[ShowInInspector]
		[InlineEditor(DrawHeader = false, ObjectFieldMode = InlineEditorObjectFieldModes.Hidden)]
		[ShowIf(nameof(albedo))]
		[BoxGroup]
		TextureData AlbedoTexture
		{
			get => albedo;
			set => albedo = value;
		}
		bool AlbedoSupported => type == MaterialType.Lambertian || type == MaterialType.Metal ||
		                        type == MaterialType.ProbabilisticVolume;
#endif

#if UNITY_EDITOR
		[AssetList]
		[ShowIf(nameof(type), MaterialType.DiffuseLight)]
#endif
		[SerializeField] TextureData emission = null;
#if UNITY_EDITOR
		[ShowInInspector]
		[InlineEditor(DrawHeader = false, ObjectFieldMode = InlineEditorObjectFieldModes.Hidden)]
		[ShowIf(nameof(emission))]
		[BoxGroup]
		TextureData EmissiveTexture
		{
			get => emission;
			set => emission = value;
		}
#endif

		public MaterialType Type => type;
		public float Roughness => roughness;
		public float RefractiveIndex => refractiveIndex;
		public Vector2 TextureScale => textureScale;
		public float Density => density;

		public TextureData Albedo
		{
			get => albedo;
			set => albedo = value;
		}

		public TextureData Emission
		{
			get => emission;
			set => emission = value;
		}

		public static MaterialData Lambertian(TextureData albedoTexture, float2 textureScale)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.Lambertian;
			data.albedo = albedoTexture;
			data.textureScale = textureScale;
			return data;
		}

		public static MaterialData Metal(TextureData albedoTexture, float2 textureScale, float fuzz = 0)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.Metal;
			data.albedo = albedoTexture;
			data.textureScale = textureScale;
			data.roughness = fuzz;
			return data;
		}

		public static MaterialData Dielectric(float refractiveIndex)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.Dielectric;
			data.refractiveIndex = refractiveIndex;
			return data;
		}

		public static MaterialData DiffuseLight(TextureData emissionTexture)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.DiffuseLight;
			data.emission = emissionTexture;
			return data;
		}

		bool TextureCanScale => (albedo && albedo.Type == TextureType.CheckerPattern) ||
		                        (emission && emission.Type == TextureType.CheckerPattern);

#if UNITY_EDITOR
		bool dirty;
		public bool Dirty => dirty || (albedo && albedo.Dirty) || (emission && emission.Dirty);

		public void ClearDirty()
		{
			dirty = false;
			if (albedo) albedo.ClearDirty();
			if (emission) emission.ClearDirty();
		}

		void OnValidate()
		{
			dirty = true;
		}
#endif
	}
}