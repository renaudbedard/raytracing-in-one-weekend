using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

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

		[SerializeField] Vector2 textureScale = Vector2.one;

		[ShowIf(nameof(Type), MaterialType.Metal)]
		[Range(0, 1)] [SerializeField] float fuzz = 0;

		[ShowIf(nameof(Type), MaterialType.Dielectric)]
		[Range(1, 2.65f)] [SerializeField] float refractiveIndex = 1;

#if UNITY_EDITOR
		[AssetList]
		[HideLabel]
		[ShowIf(nameof(AlbedoSupported))]
		[BoxGroup("Albedo")]
#endif
		[SerializeField] TextureData albedo = null;
#if UNITY_EDITOR
		[ShowInInspector]
		[InlineEditor(DrawHeader = false, ObjectFieldMode = InlineEditorObjectFieldModes.Hidden)]
		[ShowIf(nameof(albedo))]
		[BoxGroup("Albedo")]
		TextureData AlbedoTexture
		{
			get => albedo;
			set => albedo = value;
		}
		bool AlbedoSupported => type == MaterialType.Lambertian || type == MaterialType.Metal;
#endif

#if UNITY_EDITOR
		[AssetList]
		[HideLabel]
		[ShowIf(nameof(type), MaterialType.DiffuseLight)]
		[BoxGroup("Emission")]
#endif
		[SerializeField] TextureData emission = null;
#if UNITY_EDITOR
		[ShowInInspector]
		[InlineEditor(DrawHeader = false, ObjectFieldMode = InlineEditorObjectFieldModes.Hidden)]
		[ShowIf(nameof(emission))]
		[BoxGroup("Emission")]
		TextureData EmissiveTexture
		{
			get => emission;
			set => emission = value;
		}
#endif

		public MaterialType Type => type;
		public float Fuzz => fuzz;
		public float RefractiveIndex => refractiveIndex;
		public Vector2 TextureScale => textureScale;

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
			data.fuzz = fuzz;
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