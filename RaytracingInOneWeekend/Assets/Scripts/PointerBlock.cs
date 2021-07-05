using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Assertions;

namespace RaytracerInOneWeekend
{
	// TODO: Seperate head block from other blocks, keep tail block pointer around
	public unsafe struct PointerBlock<T> where T : unmanaged
	{
		public readonly T** Data;
		public readonly int Capacity;

		public int ChainLength;
		public PointerBlock<T>* NextBlock;

		public PointerBlock(T** data, int capacity) : this()
		{
			Assert.IsTrue(capacity >= 1, "Capacity must be at least 1");
			Capacity = capacity;
			Data = data;
		}

		bool TrySet(int index, T* value, out PointerBlock<T>* parentBlock)
		{
			parentBlock = null;
			if (index < Capacity)
			{
				Data[index] = value;
				return true;
			}
			if (NextBlock == null) return false;
			index -= Capacity;
			parentBlock = NextBlock;

			while (true)
			{
				if (index < parentBlock->Capacity)
				{
					parentBlock->Data[index] = value;
					return true;
				}

				if (parentBlock->NextBlock == null)
					return false;

				index -= parentBlock->Capacity;
				parentBlock = parentBlock->NextBlock;
			}
		}

		public T* Head => Data[0];

		public T* Tail
		{
			get
			{
				var parentBlock = this;
				int tailIndex = ChainLength - 1;
				while (tailIndex >= parentBlock.Capacity)
				{
					Assert.IsFalse(parentBlock.NextBlock == null, "No more blocks!");
					tailIndex -= parentBlock.Capacity;
					parentBlock = *parentBlock.NextBlock;
				}
				return parentBlock.Data[tailIndex];
			}
		}

		public bool TryPush(T* value, out PointerBlock<T>* parentBlock)
		{
			if (!TrySet(ChainLength, value, out parentBlock))
				return false;

			ChainLength++;
			return true;
		}

		public T* Pop()
		{
			Assert.IsTrue(ChainLength > 0, "Nothing to pop!");
			var tailValue = Tail;
			ChainLength--;
			return tailValue;
		}
	}
}