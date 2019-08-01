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
		[SerializeField] Color mainColor;

		[ShowIf(nameof(Type), TextureType.CheckerPattern)]
		[SerializeField] Color secondaryColor;

		[ShowIf(nameof(Type), TextureType.PerlinNoise)]
		[SerializeField] float noiseFrequency = 1;

		public TextureType Type => type;
		public Color MainColor => mainColor;
		public Color SecondaryColor => secondaryColor;
		public float NoiseFrequency => noiseFrequency;

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

		// TODO: factory method for perlin noise

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
}