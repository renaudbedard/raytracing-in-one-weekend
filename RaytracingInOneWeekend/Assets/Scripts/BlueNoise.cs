using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	class BlueNoiseData
	{
		readonly Texture2D[] noiseTextures;
		int textureIndex;

		public BlueNoiseData(Texture2D[] noiseTextures)
		{
			this.noiseTextures = noiseTextures;
		}

		public unsafe BlueNoise GetNoiseData(uint seed)
		{
			return new BlueNoise(seed,
				(half4*) noiseTextures[textureIndex].GetRawTextureData<RGBX64>().GetUnsafeReadOnlyPtr());
		}

		public void IncrementIndex()
		{
			textureIndex = (textureIndex + 1) % noiseTextures.Length;
		}
	}

	unsafe struct BlueNoise
	{
		[NativeDisableUnsafePtrRestriction] readonly half4* noiseData;
		uint2 offset;
		uint n;

		public BlueNoise(uint seed, half4* noiseData) : this()
		{
			this.noiseData = noiseData;
			n = seed;
			Advance();
		}

		public half3 NextHalf3(uint2 coordinates)
		{
			coordinates = (coordinates + offset) % 256;
			half3 value = noiseData[coordinates.y * 256 + coordinates.x].xyz;
			Advance();
			return value;
		}

		public half NextHalf(uint2 coordinates)
		{
			coordinates = (coordinates + offset) % 256;
			half value = noiseData[coordinates.y * 256 + coordinates.x].x;
			Advance();
			return value;
		}

		void Advance()
		{
			offset += (uint2) floor(R2(n++) * 256);
		}

		static float2 R2(uint n)
		{
			// from : http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
			const float g = 1.32471795724474602596f;
			const float a1 = 1.0f / g;
			const float a2 = 1.0f / (g * g);

			return float2((0.5f + a1 * n) % 1, (0.5f + a2 * n) % 1);
		}
	}
}