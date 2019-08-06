using Unity.Collections.LowLevel.Unsafe;
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
	class TextureData : ScriptableObject
	{
		[SerializeField] TextureType type;

		[ColorUsage(false, true)]
		[SerializeField] Color mainColor;

		[ColorUsage(false, true)]
		[ShowIf(nameof(Type), TextureType.CheckerPattern)]
		[SerializeField] Color secondaryColor;

		[ShowIf(nameof(Type), TextureType.PerlinNoise)]
		[SerializeField] float noiseFrequency = 1;

		[ShowIf(nameof(Type), TextureType.Image)]
		[SerializeField] Texture2D image = null;

		public TextureType Type => type;
		public Color MainColor => mainColor;
		public Color SecondaryColor => secondaryColor;
		public float NoiseFrequency => noiseFrequency;
		public Texture2D Image => image;

		public static TextureData Constant(float3 color)
		{
			var data = CreateInstance<TextureData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = TextureType.Constant;
			data.mainColor = color.ToColor();
			return data;
		}

		public static TextureData CheckerPattern(float3 oddColor, float3 evenColor)
		{
			var data = CreateInstance<TextureData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = TextureType.CheckerPattern;
			data.mainColor = oddColor.ToColor();
			data.secondaryColor = evenColor.ToColor();
			return data;
		}

		// TODO: other factory methods as needed

#if UNITY_EDITOR
		public bool Dirty { get; private set; }

		public void ClearDirty()
		{
			Dirty = false;
		}

		void OnValidate()
		{
			Dirty = true;
		}
#endif
	}

	static class TextureDataExtensions
	{
		// this could be an instance method but it's kinda nice to be able to null-check
		public static unsafe Texture GetRuntimeData(this TextureData t)
		{
			bool hasImage = t != null && t.Image;
			return t
				? new Texture(t.Type, t.MainColor.ToFloat3(), t.SecondaryColor.ToFloat3(), t.NoiseFrequency,
					hasImage ? (byte*) t.Image.GetRawTextureData<RGB24>().GetUnsafeReadOnlyPtr() : null,
					hasImage ? t.Image.width : -1, hasImage ? t.Image.height : -1)
				: default;
		}
	}
}