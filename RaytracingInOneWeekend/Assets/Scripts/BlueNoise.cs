using System;
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
		public Texture2D[] NoiseTextures = null;

		int textureIndex;

		public unsafe void Linearize()
		{
			foreach (var texture in NoiseTextures)
			{
				var pixelData = (half4*) texture.GetPixelData<half4>(0).GetUnsafePtr();

				// test for linearization marker
				if (pixelData[0].w < 0.5f)
					continue;

				// set the marker
				pixelData[0].w = half(0);

				for (int i = 0; i < texture.height; i++)
				for (int j = 0; j < texture.width; j++)
					pixelData[i * texture.width + j] = (half4) pow(pixelData[i * texture.width + j], 1 / 2.2f);
			}
		}

		public unsafe BlueNoiseRuntimeData GetRuntimeData(uint seed)
		{
			if (NoiseTextures.Length == 0)
				throw new InvalidOperationException();

			textureIndex %= NoiseTextures.Length;

			Texture2D currentTexture = NoiseTextures[textureIndex];

			return new BlueNoiseRuntimeData(seed,
				(half4*) currentTexture.GetPixelData<half4>(0).GetUnsafeReadOnlyPtr(),
				(uint) currentTexture.width);
		}

		public void CycleTexture()
		{
			textureIndex = (textureIndex + 1) % NoiseTextures.Length;
		}
	}

	unsafe struct BlueNoiseRuntimeData
	{
		[field: NativeDisableUnsafePtrRestriction]
		public half4* NoiseData { get; }
		public uint RowStride { get; }

		readonly uint seed;

		public BlueNoiseRuntimeData(uint seed, half4* noiseData, uint rowStride) : this()
		{
			RowStride = rowStride;
			NoiseData = noiseData;
			this.seed = seed;
		}

		public PerPixelBlueNoise GetPerPixelData(uint2 coordinates) =>
			new PerPixelBlueNoise(seed, coordinates, NoiseData, RowStride);
	}

	unsafe struct PerPixelBlueNoise
	{
		[NativeDisableUnsafePtrRestriction] readonly half4* noiseData;
		readonly uint2 coordinates;
		uint2 offset;
		readonly uint rowStride;
		uint n;

		public PerPixelBlueNoise(uint seed, uint2 coordinates, half4* noiseData, uint rowStride) : this()
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
			half4* pPixel = noiseData + wrappedCoords.y * rowStride + wrappedCoords.x;
			Advance();
			return pPixel->x;
		}

		public float NextFloat(float min, float max) => NextFloat() * (max - min) + min;

		public float2 NextFloat2()
		{
			uint2 wrappedCoords = (coordinates + offset) % rowStride;
			half4* pPixel = noiseData + wrappedCoords.y * rowStride + wrappedCoords.x;
			Advance();
			return float2(pPixel->xy);
		}

		public float2 NextFloat2(float2 min, float2 max) => NextFloat2() * (max - min) + min;

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