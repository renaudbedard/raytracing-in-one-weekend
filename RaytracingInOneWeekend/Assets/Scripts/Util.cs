using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	static class Util
	{
		public static void Swap<T>(ref T lhs, ref T rhs) where T : struct
		{
			T temp = lhs;
			lhs = rhs;
			rhs = temp;
		}

		public static bool EnsureCapacity<T>(this ref NativeArray<T> array, int size) where T : struct
		{
			if (array.IsCreated && array.Length == size)
				return false;

			array.SafeDispose();
			array = new NativeArray<T>(size, Allocator.Persistent);
			return true;
		}

		public static void EnsureCapacity<T>(this ref NativeList<T> list, int size) where T : struct
		{
			if (!list.IsCreated)
			{
				list = new NativeList<T>(size, Allocator.Persistent);
				return;
			}

			if (list.Length < size) list.Capacity = size;
		}

		public static void SafeDispose<T>(this ref NativeArray<T> array) where T : struct
		{
			if (array.IsCreated) array.Dispose();
		}

		public static void SafeDispose<T>(this ref NativeList<T> list) where T : struct
		{
			if (list.IsCreated) list.Dispose();
		}

		public static unsafe void AddNoResize<T>(this NativeList<T> list, T element) where T : unmanaged
		{
			((UnsafeList*) NativeListUnsafeUtility.GetInternalListDataPtrUnchecked(ref list))->AddRangeNoResize<T>(&element, 1);
		}
	}
}