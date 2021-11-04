using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;

namespace Util
{
	static class Tools
	{
		public static float3 SphericalToCartesian(float theta, float phi)
		{
			sincos(float2(theta, phi), out float2 sinThetaPhi, out float2 cosThetaPhi);
			return float3(sinThetaPhi.x * cosThetaPhi.y, cosThetaPhi.x, sinThetaPhi.x * sinThetaPhi.y);
		}

		public static void GetOrthonormalBasis(float3 normal, out float3 tangent, out float3 bitangent)
		{
			// Corrected Frisvad method
			// From listing 3 in : https://graphics.pixar.com/library/OrthonormalB/paper.pdf
			float s = normal.z >= 0 ? 1.0f : -1.0f;
			float a = -1 / (s + normal.z);
			float b = normal.x * normal.y * a;
			tangent = float3(1 + s * normal.x * normal.x * a, s * b, -s * normal.x);
			bitangent = float3(b, s + normal.y * normal.y * a, -normal.y);
		}

		public static float3 TangentToWorldSpace(float3 tangentSpaceVector, float3 normal)
		{
			GetOrthonormalBasis(normal, out float3 tangent, out float3 bitangent);

			float3x3 orthogonalMatrix = float3x3(tangent, normal, bitangent);
			float3 result = mul(orthogonalMatrix, tangentSpaceVector);
			return normalize(result);
		}

		public static float3 WorldToTangentSpace(float3 worldSpaceVector, float3 normal)
		{
			GetOrthonormalBasis(normal, out float3 tangent, out float3 bitangent);

			float3x3 orthogonalMatrix = float3x3(tangent, normal, bitangent);
			float3 result = mul(transpose(orthogonalMatrix), worldSpaceVector);
			return normalize(result);
		}

		public static bool EnsureCapacity<T>(this ref NativeArray<T> array, int size) where T : struct
		{
			if (array.IsCreated && array.Length == size)
				return false;

			array.SafeDispose();
			array = new NativeArray<T>(size, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
			return true;
		}

		public static void EnsureCapacity<T>(this ref NativeList<T> list, int size) where T : unmanaged
		{
			if (!list.IsCreated)
			{
				list = new NativeList<T>(size, Allocator.Persistent);
				return;
			}

			if (list.Length < size) list.Capacity = size;
		}

		public static void EnsureCapacity<T>(this ref UnsafePtrList<T> list, int size) where T : unmanaged
		{
			if (!list.IsCreated)
			{
				list = new UnsafePtrList<T>(size, Allocator.Persistent);
				return;
			}

			if (list.Length < size) list.Capacity = size;
		}

		public static void SafeDispose<T>(this ref NativeArray<T> array) where T : struct
		{
			if (array.IsCreated) array.Dispose();
		}

		public static unsafe void ZeroMemory<T>(this ref NativeArray<T> array) where T : unmanaged
		{
			UnsafeUtility.MemClear(array.GetUnsafePtr(), array.Length * sizeof(T));
		}

		public static void SafeDispose<T>(this ref NativeList<T> list) where T : unmanaged
		{
			if (list.IsCreated) list.Dispose();
		}

		public static T PeekOrDefault<T>(this Queue<T> queue)
		{
			if (queue.Count > 0) return queue.Peek();
			return default;
		}

		public static IEnumerable<int> SpaceFillingSeries(int length)
		{
			int current = 0;
			var seen = new HashSet<int>();
			do
			{
				int divider = 2;
				do
				{
					int increment = (int) ceil((float) length / divider);
					for (int i = 0; i < divider; i++)
					{
						current = i * increment;
						if (!seen.Contains(current))
							break;
					}

					divider *= 2;
				} while (seen.Contains(current));

				yield return current;
				seen.Add(current);
			} while (seen.Count < length);
		}

		public static unsafe T* Pop<T>(this ref UnsafePtrList<T> list) where T : unmanaged
		{
			T* element = list[^1];
			list.Resize(list.Length - 1);
			return element;
		}

		public static bool TryGetProperty(this Material material, string name, out float value)
		{
			if (!material.HasFloat(name))
			{
				value = default;
				return false;
			}
			value = material.GetFloat(name);
			return true;
		}

		public static bool TryGetProperty(this Material material, string name, out Texture2D value)
		{
			if (!material.HasTexture(name))
			{
				value = default;
				return false;
			}
			value = material.GetTexture(name) as Texture2D;
			return value != null;
		}

		public static bool TryGetProperty(this Material material, string name, out Cubemap value)
		{
			if (!material.HasTexture(name))
			{
				value = default;
				return false;
			}
			value = material.GetTexture(name) as Cubemap;
			return value != null;
		}

		public static bool TryGetProperty(this Material material, string name, out int value)
		{
			if (!material.HasInt(name))
			{
				value = default;
				return false;
			}
			value = material.GetInt(name);
			return true;
		}

		public static bool TryGetProperty(this Material material, string name, out Color value)
		{
			if (!material.HasColor(name))
			{
				value = default;
				return false;
			}
			value = material.GetColor(name);
			return true;
		}

		public static float LinearToGamma(float value)
		{
			value = max(value, 0);
			return max(1.055f * pow(value, 0.416666667f) - 0.055f, 0);
		}

		// ACES fitted implementation by Stephen Hill (@self_shadow)
		// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl

		// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
		static readonly float3x3 ACESInputMat = float3x3(
			0.59719f, 0.35458f, 0.04823f,
			0.07600f, 0.90834f, 0.01566f,
			0.02840f, 0.13383f, 0.83777f);

		// ODT_SAT => XYZ => D60_2_D65 => sRGB
		static readonly float3x3 ACESOutputMat = float3x3(
			1.60475f, -0.53108f, -0.07367f,
			-0.10208f, 1.10813f, -0.00605f,
			-0.00327f, -0.07276f, 1.07602f
		);

		static float3 RRTAndODTFit(float3 v)
		{
			float3 a = v * (v + 0.0245786f) - 0.000090537f;
			float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
			return a / b;
		}

		public static float3 ACESFitted(float3 color)
		{
			color = mul(ACESInputMat, color);
			color = RRTAndODTFit(color);
			color = mul(ACESOutputMat, color);
			color = saturate(color);
			return color;
		}

		public static float3 ACESFilm(float3 x)
		{
			const float a = 2.51f;
			const float b = 0.03f;
			const float c = 2.43f;
			const float d = 0.59f;
			const float e = 0.14f;
			return saturate((x*(a*x+b))/(x*(c*x+d)+e));
		}
	}
}