using System;
using Runtime;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace Unity
{
	[Serializable]
	class BlueNoiseData
	{
		[LabelText("Textures")] [AssetsOnly] public Texture2D[] NoiseTextures = null;

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

		public unsafe BlueNoise GetRuntimeData(uint seed)
		{
			if (NoiseTextures.Length == 0)
				throw new InvalidOperationException();

			textureIndex %= NoiseTextures.Length;

			Texture2D currentTexture = NoiseTextures[textureIndex];

			return new BlueNoise(seed,
				(half4*) currentTexture.GetPixelData<half4>(0).GetUnsafeReadOnlyPtr(),
				(uint) currentTexture.width);
		}

		public void CycleTexture()
		{
			textureIndex = (textureIndex + 1) % NoiseTextures.Length;
		}
	}
}