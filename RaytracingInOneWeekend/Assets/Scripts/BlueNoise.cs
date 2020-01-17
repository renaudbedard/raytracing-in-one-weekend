using System;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif


namespace RaytracerInOneWeekend
{
	[Serializable]
	class BlueNoise
	{
		[LabelText("Textures")] [AssetsOnly]
		public Texture2D[] NoiseTextures;

		int textureIndex;

		public unsafe BlueNoiseRuntimeData GetRuntimeData(uint seed)
		{
			if (NoiseTextures.Length == 0)
				throw new InvalidOperationException();

			textureIndex %= NoiseTextures.Length;

			Texture2D currentTexture = NoiseTextures[textureIndex];

			return new BlueNoiseRuntimeData(seed,
				(half*) currentTexture.GetPixelData<half>(0).GetUnsafeReadOnlyPtr(),
				(ushort) currentTexture.width);
		}

		public void CycleTexture()
		{
			textureIndex = (textureIndex + 1) % NoiseTextures.Length;
		}
	}

	unsafe struct BlueNoiseRuntimeData
	{
		[NativeDisableUnsafePtrRestriction] readonly half* noiseData;
		readonly ushort rowStride;
		readonly uint seed;

		public BlueNoiseRuntimeData(uint seed, half* noiseData, ushort rowStride) : this()
		{
			this.noiseData = noiseData;
			this.rowStride = rowStride;
			this.seed = seed;
		}

		public PerPixelBlueNoise GetPerPixelData(uint2 coordinates) =>
			new PerPixelBlueNoise(seed, coordinates, noiseData, rowStride);
	}

	unsafe struct PerPixelBlueNoise
	{
		[NativeDisableUnsafePtrRestriction] readonly half* noiseData;
		readonly uint2 coordinates;
		uint2 offset;
		readonly uint rowStride;
		uint n;

		public PerPixelBlueNoise(uint seed, uint2 coordinates, half* noiseData, ushort rowStride) : this()
		{
			this.coordinates = coordinates;
			this.noiseData = noiseData;
			this.rowStride = rowStride;

			n = (byte) (seed % 255);
			Advance();
		}

		public float NextFloat()
		{
			uint2 wrappedCoords = (coordinates + offset) % rowStride;
			half* pPixel = noiseData + wrappedCoords.y * rowStride * 4 + wrappedCoords.x * 4;
			Advance();
			return pPixel[0];
		}

		public float NextFloat(float min, float max) => NextFloat() * (max - min) + min;

		public float2 NextFloat2()
		{
			uint2 wrappedCoords = (coordinates + offset) % rowStride;
			half* pPixel = noiseData + wrappedCoords.y * rowStride * 4 + wrappedCoords.x * 4;
			Advance();
			return float2(pPixel[0], pPixel[1]);
		}

		public float2 NextFloat2(float2 min, float2 max) => NextFloat2() * (max - min) + min;

		public float3 NextFloat3()
		{
			uint2 wrappedCoords = (coordinates + offset) % rowStride;
			half* pPixel = noiseData + wrappedCoords.y * rowStride * 4 + wrappedCoords.x * 4;
			Advance();
			return float3(pPixel[0], pPixel[1], pPixel[2]);
		}

		public int NextInt(int min, int max) => (int) floor(NextFloat() * (max - min) + min);

		void Advance()
		{
			offset += (uint2) floor(R2(n++) * rowStride);
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