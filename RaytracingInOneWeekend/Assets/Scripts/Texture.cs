using JetBrains.Annotations;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	enum TextureType
	{
		None,
		Constant,
		CheckerPattern
	}

	struct Texture
	{
		public readonly TextureType Type;
		public readonly float3 MainColor;
		public readonly float3 SecondaryColor;

		public Texture(TextureType type, float3 mainColor, float3 secondaryColor)
		{
			Type = type;
			MainColor = mainColor;
			SecondaryColor = secondaryColor;
		}

		[Pure]
		public float3 Value(float3 position)
		{
			switch (Type)
			{
				case TextureType.Constant:
					return MainColor;

				case TextureType.CheckerPattern:
					float3 sines = sin(10 * position);
					return sines.x * sines.y * sines.z < 0 ? MainColor : SecondaryColor;
			}

			return default;
		}
	}
}