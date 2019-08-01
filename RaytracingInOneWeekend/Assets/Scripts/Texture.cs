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
		public readonly float NoiseFrequency;

		public Texture(TextureType type, float3 mainColor, float3 secondaryColor, float noiseFrequency)
		{
			Type = type;
			MainColor = mainColor;
			SecondaryColor = secondaryColor;
			NoiseFrequency = noiseFrequency;
		}

		[Pure]
		public float3 Value(float3 position, float3 normal, float2 scale, PerlinData perlinData)
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
					return perlinData.Sample(position * NoiseFrequency);
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
		[NativeDisableUnsafePtrRestriction] readonly int* permX, permY, permZ;
		[NativeDisableUnsafePtrRestriction] readonly float3* randomVectors;

		public PerlinData(int* permX, int* permY, int* permZ, float3* randomVectors)
		{
			this.permX = permX;
			this.permY = permY;
			this.permZ = permZ;
			this.randomVectors = randomVectors;
		}

		[Pure]
		public float Sample(float3 position)
		{
			float3 uvw = smoothstep(0, 1, frac(position));
			var ijk = (int3) floor(position);

			float3* samples = stackalloc float3[2 * 2 * 2];
			float3* sampleCursor = samples;

			for (int di = 0; di < 2; di++)
			for (int dj = 0; dj < 2; dj++)
			for (int dk = 0; dk < 2; dk++)
			{
				*sampleCursor++ = randomVectors[permX[(ijk.x + di) & 255] ^ permY[(ijk.y + dj) & 255] ^ permZ[(ijk.z + dk) & 255]];
			}

			// TODO
			// float2x2 alongX = Util.Lerp(slices[0], slices[1], uvw.x);
			// float2 alongY = lerp(alongX[0], alongX[1], uvw.y);
			// return lerp(alongY[0], alongY[1], uvw.z);

			return 0;
		}
	}
}