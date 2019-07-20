using Unity.Mathematics;
using UnityEngine;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#endif

namespace RaytracerInOneWeekend
{
	[CreateAssetMenu]
	class TextureData : ScriptableObject
	{
		[SerializeField] TextureType type;
		[SerializeField] Color mainColor;

#if ODIN_INSPECTOR
		[ShowIf(nameof(Type), TextureType.CheckerPattern)]
#endif
		[SerializeField] Color secondaryColor;

		public TextureType Type => type;
		public Color MainColor => mainColor;
		public Color SecondaryColor => secondaryColor;

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