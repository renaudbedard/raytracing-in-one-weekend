using System;
using JetBrains.Annotations;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	enum TextureType
	{
		None,
		Constant,
		CheckerPattern,
		PerlinNoise,
		Image,
	}

	unsafe struct Texture
	{
		public readonly TextureType Type;
		public readonly float3 MainColor;
		public readonly float3 SecondaryColor;
		public readonly float NoiseFrequency;
		public readonly int2 ImageSize;
		public readonly byte* ImagePointer;

		public Texture(TextureType type, float3 mainColor, float3 secondaryColor, float noiseFrequency, byte* pImage,
			int imageWidth, int imageHeight)
		{
			Type = type;
			MainColor = mainColor;
			SecondaryColor = secondaryColor;
			NoiseFrequency = noiseFrequency;
			ImagePointer = pImage;
			ImageSize = int2(imageWidth, imageHeight);
		}

		[Pure]
		public float3 Value(float3 position, float3 normal, float2 scale, PerlinData perlinData)
		{
			// TODO: make this use UVW coordinates instead

			switch (Type)
			{
				case TextureType.Constant:
					return MainColor;

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
					return 0.5f * (1 + sin(NoiseFrequency * position.x +
					                       5 * perlinData.Turbulence(NoiseFrequency * position))) *
					       MainColor;

				case TextureType.Image:
				{
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

	class PerlinDataGenerator : IDisposable
	{
		const int BufferSize = 256;

		NativeArray<int> xPermBuffer, yPermBuffer, zPermBuffer;
		NativeArray<float3> randomVectorBuffer;

		public void Generate(uint seed)
		{
			var rng = new Random(seed);

			void GeneratePermutationBuffer(ref NativeArray<int> buffer)
			{
				buffer.EnsureCapacity(BufferSize);

				for (int i = 0; i < BufferSize; i++)
					buffer[i] = i;

				for (int i = BufferSize - 1; i > 0; i--)
				{
					int target = rng.NextInt(0, i + 1);
					int tmp = buffer[i];
					buffer[i] = buffer[target];
					buffer[target] = tmp;
				}
			}

			GeneratePermutationBuffer(ref xPermBuffer);
			GeneratePermutationBuffer(ref yPermBuffer);
			GeneratePermutationBuffer(ref zPermBuffer);

			randomVectorBuffer.EnsureCapacity(BufferSize);
			for (int i = 0; i < BufferSize; i++)
				randomVectorBuffer[i] = rng.NextFloat3Direction();
		}

		public void Dispose()
		{
			xPermBuffer.SafeDispose();
			yPermBuffer.SafeDispose();
			zPermBuffer.SafeDispose();
			randomVectorBuffer.SafeDispose();
		}

		public unsafe PerlinData GetRuntimeData()
		{
			return new PerlinData(
				(int*) xPermBuffer.GetUnsafeReadOnlyPtr(),
				(int*) yPermBuffer.GetUnsafeReadOnlyPtr(),
				(int*) zPermBuffer.GetUnsafeReadOnlyPtr(),
				(float3*) randomVectorBuffer.GetUnsafeReadOnlyPtr());
		}
	}

	unsafe struct PerlinData
	{
		// based on : http://www.eastfarthing.com/blog/2015-04-21-noise/

		[NativeDisableUnsafePtrRestriction] readonly int* permX, permY, permZ;
		[NativeDisableUnsafePtrRestriction] readonly float3* randomVectors;

		public PerlinData(int* permX, int* permY, int* permZ, float3* randomVectors)
		{
			this.permX = permX;
			this.permY = permY;
			this.permZ = permZ;
			this.randomVectors = randomVectors;
		}

		static float3 Falloff(float3 t)
		{
			t = abs(t);
			return select(1 - (3 - 2 * t) * t * t, 0, t >= 1);
		}

		static float Surflet(float3 fp, float3 grad)
		{
			float3 ffp = Falloff(fp);
			return ffp.x * ffp.y * ffp.z * dot(fp, grad);
		}

		[Pure]
		public float Noise(float3 position)
		{
			var cellPos = (int3) floor(position);
			float result = 0;

			for (int k = cellPos.z; k <= cellPos.z + 1; k++)
			for (int j = cellPos.y; j <= cellPos.y + 1; j++)
			for (int i = cellPos.x; i <= cellPos.x + 1; i++)
			{
				int hash = permX[i & 255] ^ permY[j & 255] ^ permZ[k & 255];
				float3 fractionalPos = position - float3(i, j, k);
				result += Surflet(fractionalPos, randomVectors[hash]);
			}

			return saturate(result);
		}

		[Pure]
		public float Turbulence(float3 position, int depth = 7)
		{
			float accumulator = 0;
			float3 p = position;
			float weight = 1;

			for (int i = 0; i < depth; i++)
			{
				accumulator += weight * Noise(p);
				weight *= 0.5f;
				p *= 2;
			}

			return saturate(accumulator);
		}
	}
}