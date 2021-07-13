using System;
using JetBrains.Annotations;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using Util;
using static Unity.Mathematics.math;

namespace Runtime
{
	enum TextureType
	{
		None,
		Constant,
		CheckerPattern,
		PerlinNoise,
		Image,
		ConstantScalar
	}

	readonly unsafe struct Texture
	{
		public readonly TextureType Type;
		public readonly float3 MainColor;
		public readonly float3 SecondaryColor;
		public readonly float Parameter;
		public readonly int2 ImageSize;
		public readonly byte* ImagePointer;

		float NoiseFrequency => Parameter;
		float ScalarValue => Parameter;

		public Texture(TextureType type, float3 mainColor, float3 secondaryColor, float parameter, byte* pImage,
			int imageWidth, int imageHeight)
		{
			Type = type;
			MainColor = mainColor;
			SecondaryColor = secondaryColor;
			Parameter = parameter;
			ImagePointer = pImage;
			ImageSize = int2(imageWidth, imageHeight);
		}

		[Pure]
		public float3 Value(float3 position, float3 normal, float2 scale, PerlinNoise perlinNoise)
		{
			// TODO: make this use UVW coordinates instead

			switch (Type)
			{
				case TextureType.Constant:
					return MainColor;

				case TextureType.ConstantScalar:
					return ScalarValue;

				case TextureType.CheckerPattern:
				{
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

				case TextureType.PerlinNoise:
					return 0.5f * (1 + sin(NoiseFrequency * position.z +
					                       10 * perlinNoise.Turbulence(position))) *
					       MainColor;

				case TextureType.Image:
				{
					if (ImagePointer == null)
						return 0;

					float phi = atan2(normal.z, normal.x);
					float theta = asin(normal.y);
					float2 uv = float2((phi + PI) / (2 * PI), (theta + PI / 2) / PI);
					int2 coords = (int2) (uv * ImageSize);

					byte* pPixelData = ImagePointer + (coords.y * ImageSize.x + coords.x) * 3;
					return float3(pPixelData[0], pPixelData[1], pPixelData[2]) / 255 * MainColor;
				}
			}

			return default;
		}
	}

	readonly unsafe struct Cubemap
	{
		public readonly int2 FaceSize;
		public readonly ChannelType ChannelType;
		public readonly int PixelStride;

		[NativeDisableUnsafePtrRestriction]
		public readonly byte* NegativeX, PositiveX, NegativeY, PositiveY, NegativeZ, PositiveZ;

		public Cubemap(UnityEngine.Cubemap cubemap)
		{
			FaceSize = int2(cubemap.width, cubemap.height);

			switch (cubemap.graphicsFormat)
			{
				case GraphicsFormat.R16G16B16A16_SFloat:
					ChannelType = ChannelType.SignedHalf;
					PixelStride = (4 * 16) / 8;
					break;

				default:
					throw new NotSupportedException();
			}

			NegativeX = (byte*) cubemap.GetPixelData<byte>(0, CubemapFace.NegativeX).GetUnsafeReadOnlyPtr();
			PositiveX = (byte*) cubemap.GetPixelData<byte>(0, CubemapFace.PositiveX).GetUnsafeReadOnlyPtr();
			NegativeY = (byte*) cubemap.GetPixelData<byte>(0, CubemapFace.NegativeY).GetUnsafeReadOnlyPtr();
			PositiveY = (byte*) cubemap.GetPixelData<byte>(0, CubemapFace.PositiveY).GetUnsafeReadOnlyPtr();
			NegativeZ = (byte*) cubemap.GetPixelData<byte>(0, CubemapFace.NegativeZ).GetUnsafeReadOnlyPtr();
			PositiveZ = (byte*) cubemap.GetPixelData<byte>(0, CubemapFace.PositiveZ).GetUnsafeReadOnlyPtr();
		}

		[Pure]
		public float3 Sample(float3 vector)
		{
			if (NegativeX == (byte*) 0)
				return default;

			// indexing math adapted from : https://scalibq.wordpress.com/2013/06/23/cubemaps/
			float4 absVector = float4(abs(vector), 0);
			float maxDistance = cmax(absVector);
			int laneMask = bitmask(maxDistance == absVector);
			int firstLane = tzcnt(laneMask);
			bool positive = vector[firstLane] >= 0;

			float u, v;
			int2 coords;
			byte* pFaceData;

			switch (firstLane)
			{
				case 0: // x
					u = (((positive ? -vector.z : vector.z) / absVector.x) + 1) / 2;
					v = ((-vector.y / absVector.x) + 1) / 2;

					coords = (int2) (float2(u, v) * FaceSize);
					pFaceData = positive ? PositiveX : NegativeX;
					break;

				case 1: // y
					u = ((vector.x / absVector.y) + 1) / 2;
					v = (((positive ? vector.z : -vector.z) / absVector.y) + 1) / 2;

					coords = (int2) (float2(u, v) * FaceSize);
					pFaceData = positive ? PositiveY : NegativeY;
					break;

				case 2: // z
					u = (((positive ? vector.x : -vector.x) / absVector.z) + 1) / 2;
					v = ((-vector.y / absVector.z) + 1) / 2;

					coords = (int2) (float2(u, v) * FaceSize);
					pFaceData = positive ? PositiveZ : NegativeZ;
					break;

				default:
					throw new InvalidOperationException();
			}

			pFaceData += coords.y * FaceSize.x * PixelStride + coords.x * PixelStride;

			switch (ChannelType)
			{
				case ChannelType.UnsignedByte:
					return float3(pFaceData[0], pFaceData[1], pFaceData[2]) / 255;

				case ChannelType.SignedHalf:
					var pTypedData = (half*) pFaceData;
					return float3(pTypedData[0], pTypedData[1], pTypedData[2]);

				default:
					throw new InvalidOperationException();
			}
		}
	}
}