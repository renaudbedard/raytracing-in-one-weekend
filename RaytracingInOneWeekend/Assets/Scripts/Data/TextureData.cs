using System;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace RaytracerInOneWeekend
{
	[Serializable]
	struct TextureData
	{
		[SerializeField] TextureType type;

		[ColorUsage(false, true)]
		[SerializeField] Color mainColor;

		[ColorUsage(false, true)]
		[ShowIf(nameof(Type), TextureType.CheckerPattern)]
		[SerializeField] Color secondaryColor;

		[ShowIf(nameof(Type), TextureType.PerlinNoise)]
		[SerializeField] float noiseFrequency;

		[ShowIf(nameof(Type), TextureType.Image)]
		[SerializeField] Texture2D image;

		public TextureType Type => type;
		public Color MainColor => mainColor;
		public Color SecondaryColor => secondaryColor;
		public float NoiseFrequency => noiseFrequency;
		public Texture2D Image => image;

		public static TextureData Constant(float3 color)
		{
			return new TextureData { type = TextureType.Constant, mainColor = color.ToColor() };
		}

		public static TextureData CheckerPattern(float3 oddColor, float3 evenColor)
		{
			return new TextureData
			{
				type = TextureType.CheckerPattern,
				mainColor = oddColor.ToColor(),
				secondaryColor = evenColor.ToColor()
			};
		}

		public unsafe Texture GetRuntimeData()
		{
			switch (Type)
			{
				case TextureType.Image:
					// TODO: adapt to texture format
					return new Texture(Type, MainColor.ToFloat3(), SecondaryColor.ToFloat3(), NoiseFrequency,
						Image ? (byte*) Image.GetRawTextureData<RGB24>().GetUnsafeReadOnlyPtr() : null,
						Image ? Image.width : -1, Image ? Image.height : -1);

				default:
					return new Texture(Type, MainColor.ToFloat3(), SecondaryColor.ToFloat3(), NoiseFrequency, null, -1, -1);
			}
		}

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