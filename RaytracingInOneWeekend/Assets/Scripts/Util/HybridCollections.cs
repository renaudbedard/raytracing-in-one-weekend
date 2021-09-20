using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using static Unity.Mathematics.math;

namespace Util
{
	public unsafe struct HybridPtrStack<T> where T : unmanaged
	{
		public int Length { get; private set; }

		T** buffer;
		int capacity;

		public HybridPtrStack(T** stackBuffer, int stackCapacity)
		{
			buffer = stackBuffer;
			capacity = stackCapacity;
			Length = 0;
		}

		void ReallocateHeap(int toCapacity)
		{
			var newBuffer = (T**) UnsafeUtility.Malloc(sizeof(T*) * toCapacity, sizeof(AlignHelper) - sizeof(T*), Allocator.Temp);
			UnsafeUtility.MemCpy(newBuffer, buffer, sizeof(T*) * Length);
			buffer = newBuffer;
			capacity = toCapacity;
		}

		public void Push(T* element)
		{
			if (Length >= capacity)
				ReallocateHeap((int) ceil(capacity * 1.5f));

			buffer[Length++] = element;
		}

		public T* Pop() => buffer[--Length];

		public void Clear() => Length = 0;

		public T* this[int i] => buffer[i];

		struct AlignHelper
		{
			public byte dummy;
			public T* data;
		}
	}

	public unsafe struct HybridList<T> where T : unmanaged
	{
		public int Length { get; private set; }

		T* buffer;
		int capacity;

		public HybridList(T* stackBuffer, int stackCapacity)
		{
			buffer = stackBuffer;
			capacity = stackCapacity;
			Length = 0;
		}

		void ReallocateHeap(int toCapacity)
		{
			var newBuffer = (T*) UnsafeUtility.Malloc(sizeof(T) * toCapacity, UnsafeUtility.AlignOf<T>(), Allocator.Temp);
			UnsafeUtility.MemCpy(newBuffer, buffer, sizeof(T) * Length);
			buffer = newBuffer;
			capacity = toCapacity;
		}

		public void Add(T element)
		{
			if (Length >= capacity)
				ReallocateHeap((int) ceil(capacity * 1.5f));

			buffer[Length++] = element;
		}

		public void Clear() => Length = 0;

		public T this[int i] => buffer[i];

		public void Sort<U>(U comparer) where U : IComparer<T> => NativeSortExtension.Sort(buffer, Length, comparer);
	}
}