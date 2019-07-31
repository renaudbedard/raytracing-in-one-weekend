using System;
using JetBrains.Annotations;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	enum TextureType
	{
		None,
		Constant,
		CheckerPattern,
		PerlinNoise
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
		public unsafe float3 Value(float3 position, float3 normal, float2 scale, PerlinData perlinData)
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

				case TextureType.PerlinNoise:
					return perlinData.Sample(position);
			}

			return default;
		}
	}

	class PerlinDataGenerator : IDisposable
	{
		const int BufferSize = 256;

		NativeArray<int> xPermBuffer, yPermBuffer, zPermBuffer;
		NativeArray<float> randomFloatBuffer;

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

			randomFloatBuffer.EnsureCapacity(BufferSize);
			for (int i = 0; i < BufferSize; i++)
				randomFloatBuffer[i] = rng.NextFloat();
		}

		public void Dispose()
		{
			xPermBuffer.SafeDispose();
			yPermBuffer.SafeDispose();
			zPermBuffer.SafeDispose();
			randomFloatBuffer.SafeDispose();
		}

		public unsafe PerlinData GetRuntimeData()
		{
			return new PerlinData(
				(int*) xPermBuffer.GetUnsafeReadOnlyPtr(),
				(int*) yPermBuffer.GetUnsafeReadOnlyPtr(),
				(int*) zPermBuffer.GetUnsafeReadOnlyPtr(),
				(float*) randomFloatBuffer.GetUnsafeReadOnlyPtr());
		}
	}

	unsafe struct PerlinData
	{
		[NativeDisableUnsafePtrRestriction] readonly int* permX, permY, permZ;
		[NativeDisableUnsafePtrRestriction] readonly float* randomFloats;

		public PerlinData(int* permX, int* permY, int* permZ, float* randomFloats)
		{
			this.permX = permX;
			this.permY = permY;
			this.permZ = permZ;
			this.randomFloats = randomFloats;
		}

		[Pure]
		public float Sample(float3 position)
		{
			int3 ijk = int3(4 * position) & 255;
			return randomFloats[permX[ijk.x] ^ permY[ijk.y] ^ permZ[ijk.z]];
		}
	}
}