using System;
using Runtime;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Util;
using Random = Unity.Mathematics.Random;
using static Unity.Mathematics.math;

namespace Unity
{
	class PerlinNoiseData : IDisposable
	{
		const int BufferSize = 256; // must be a square

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

			int sqrtBufferSize = (int) sqrt(BufferSize);

			// uniform normalized vectors on a sphere
			// adapted from https://medium.com/@all2one/generating-uniformly-distributed-points-on-sphere-1f7125978c4c#db7c
			int index = 0;
			for (int i = 0; i < sqrtBufferSize; i++)
			{
				float z = (i + 0.5f) / sqrtBufferSize * 2 - 1;

				for (int j = 0; j < sqrtBufferSize; j++)
				{
					float t = (float) j / sqrtBufferSize * 2 * PI;
					float r = sqrt(1 - z * z);
					sincos(t, out float sinT, out float cosT);

					randomVectorBuffer[index++] = float3(r * cosT, r * sinT, z);
				}
			}
		}

		public void Dispose()
		{
			xPermBuffer.SafeDispose();
			yPermBuffer.SafeDispose();
			zPermBuffer.SafeDispose();
			randomVectorBuffer.SafeDispose();
		}

		public unsafe PerlinNoise GetRuntimeData()
		{
			return new PerlinNoise(
				(int*) xPermBuffer.GetUnsafeReadOnlyPtr(),
				(int*) yPermBuffer.GetUnsafeReadOnlyPtr(),
				(int*) zPermBuffer.GetUnsafeReadOnlyPtr(),
				(float3*) randomVectorBuffer.GetUnsafeReadOnlyPtr());
		}
	}
}