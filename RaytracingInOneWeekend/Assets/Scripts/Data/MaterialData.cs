using Unity.Mathematics;
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
		[SerializeField] TextureData albedo = null;

		[SerializeField] Vector2 textureScale = Vector2.one;

		[ShowIf(nameof(Type), MaterialType.Metal)]
		[Range(0, 1)] [SerializeField] float fuzz = 0;

		[ShowIf(nameof(Type), MaterialType.Dielectric)]
		[Range(1, 2.65f)] [SerializeField] float refractiveIndex = 1;

		public MaterialType Type => type;
		public float Fuzz => fuzz;
		public float RefractiveIndex => refractiveIndex;
		public Vector2 TextureScale => textureScale;

		public TextureData Albedo
		{
			get => albedo;
			set => albedo = value;
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
		public bool Dirty => dirty || (albedo && albedo.Dirty);

		public void ClearDirty()
		{
			dirty = false;
			if (albedo) albedo.ClearDirty();
		}

		void OnValidate()
		{
			dirty = true;
		}
#endif
	}
}