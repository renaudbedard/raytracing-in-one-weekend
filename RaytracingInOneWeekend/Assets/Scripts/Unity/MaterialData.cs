using Runtime;
using Sirenix.OdinInspector;
using Unity.Mathematics;
using UnityEngine;
#if ODIN_INSPECTOR

#else
using OdinMock;
#endif

namespace Unity
{
	[CreateAssetMenu]
	class MaterialData : ScriptableObject
	{
		[SerializeField] MaterialType type = MaterialType.None;

		[ShowIf(nameof(TextureCanScale))]
		[SerializeField] Vector2 textureScale = Vector2.one;

		[ShowIf(nameof(RoughnessSupported))]
		[SerializeField] TextureData roughness;

		[ShowIf(nameof(Type), MaterialType.Dielectric)]
		[Range(1, 2.65f)] [SerializeField] float refractiveIndex = 1;

		[ShowIf(nameof(Type), MaterialType.ProbabilisticVolume)]
		[Range(0, 50)] [SerializeField] float density = 1;

#if UNITY_EDITOR
		[ShowIf(nameof(AlbedoSupported))]
#endif
		[SerializeField] TextureData albedo;

#if UNITY_EDITOR
		[ShowIf(nameof(type), MaterialType.DiffuseLight)]
#endif
		[SerializeField] TextureData emission;

#if UNITY_EDITOR
		bool AlbedoSupported => type == MaterialType.Lambertian ||
		                        type == MaterialType.Metal ||
		                        type == MaterialType.ProbabilisticVolume ||
		                        type == MaterialType.Dielectric;

		bool RoughnessSupported => type == MaterialType.Dielectric ||
		                           type == MaterialType.Metal;
#endif

		public MaterialType Type => type;
		public TextureData Roughness => roughness;
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

		public static MaterialData Metal(TextureData albedoTexture, float2 textureScale, TextureData fuzz)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.Metal;
			data.albedo = albedoTexture;
			data.textureScale = textureScale;
			data.roughness = fuzz;
			return data;
		}

		public static MaterialData Dielectric(float refractiveIndex, TextureData albedoTexture, TextureData roughnessTexture)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.Dielectric;
			data.refractiveIndex = refractiveIndex;
			data.albedo = albedoTexture;
			data.roughness = roughnessTexture;
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

		bool TextureCanScale => albedo.Type == TextureType.CheckerPattern ||
		                        emission.Type == TextureType.CheckerPattern ||
		                        roughness.Type == TextureType.CheckerPattern;

#if UNITY_EDITOR
		bool dirty;
		public bool Dirty => dirty || albedo.Dirty || emission.Dirty;

		public void ClearDirty()
		{
			dirty = false;
			albedo.ClearDirty();
			emission.ClearDirty();
		}

		void OnValidate()
		{
			dirty = true;
		}
#endif
	}
}