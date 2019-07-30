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
		public float3 Value(float3 normal, float2 scale)
		{
			switch (Type)
			{
				case TextureType.Constant:
					return MainColor;

				case TextureType.CheckerPattern:
					// from iq : https://www.shadertoy.com/view/ltl3D8
					float3 n = abs(normal);
					float3 v = n.x > n.y && n.x > n.z ? normal.xyz :
						n.y > n.x && n.y > n.z ? normal.yzx :
						normal.zxy;
					float2 q = v.yz / v.x;
					float2 uv = 0.5f + 0.5f * q;

					float2 sines = sin(PI * scale * uv);
					return sines.x * sines.y < 0 ? MainColor : SecondaryColor;
			}

			return default;
		}
	}
}