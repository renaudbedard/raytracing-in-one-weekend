using System;
using Runtime;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using Util;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace Unity
{
	[Serializable]
	class SpatioTemporalBlueNoiseData
	{
		[AssetsOnly] public Texture2D[] ScalarTextures = null;
		[AssetsOnly] public Texture2D[] Vector2Textures = null;
		[AssetsOnly] public Texture2D[] CosineUnitVector3Textures = null;
		[AssetsOnly] public Texture2D[] UnitVector2Textures = null;
		[AssetsOnly] public Texture2D[] UnitVector3Textures = null;

		int textureIndex = -1;

		public unsafe SpatioTemporalBlueNoise GetRuntimeData(uint seed)
		{
			uint rowStride = (uint) ScalarTextures[0].width;

			return new SpatioTemporalBlueNoise(seed,
				(byte*) ScalarTextures[textureIndex].GetPixelData<byte>(0).GetUnsafeReadOnlyPtr(),
				(RGB24*) Vector2Textures[textureIndex].GetPixelData<RGB24>(0).GetUnsafeReadOnlyPtr(),
				(RGBA32*) CosineUnitVector3Textures[textureIndex].GetPixelData<RGBA32>(0).GetUnsafeReadOnlyPtr(),
				(RGB24*) UnitVector2Textures[textureIndex].GetPixelData<RGB24>(0).GetUnsafeReadOnlyPtr(),
				(RGB24*) UnitVector3Textures[textureIndex].GetPixelData<RGB24>(0).GetUnsafeReadOnlyPtr(),
				rowStride);
		}

		public void CycleTexture()
		{
			textureIndex = (textureIndex + 1) % ScalarTextures.Length;
		}
	}
}