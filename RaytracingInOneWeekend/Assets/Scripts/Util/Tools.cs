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

		public static void Swap<T>(ref T lhs, ref T rhs) where T : struct
		{
			(lhs, rhs) = (rhs, lhs);
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
			return true;
		}

		public static bool TryGetProperty(this Material material, string name, out Cubemap value)
		{
			if (!material.HasTexture(name))
			{
				value = default;
				return false;
			}
			value = material.GetTexture(name) as Cubemap;
			return true;
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
	}
}