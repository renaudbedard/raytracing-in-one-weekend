using Unity.Collections;

namespace RaytracerInOneWeekend
{
	static class Util
	{
		public static void Swap<T>(ref T lhs, ref T rhs)
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
			list.Clear();
		}

		public static void SafeDispose<T>(this ref NativeArray<T> array) where T : struct
		{
			if (array.IsCreated) array.Dispose();
		}

		public static void SafeDispose<T>(this ref NativeList<T> list) where T : struct
		{
			if (list.IsCreated) list.Dispose();
		}
	}
}