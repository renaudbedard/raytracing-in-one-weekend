using static Unity.Mathematics.math;

namespace Util
{
	public unsafe struct PointerBlockChain<T> where T : unmanaged
	{
		public int Length { get; private set; }

		readonly PointerBlock<T>* headBlock;

		PointerBlock<T>* tailBlock;
		T** tail;

		public PointerBlockChain(PointerBlock<T>* firstBlock) : this()
		{
			headBlock = firstBlock;
			Clear();
		}

		public void Clear()
		{
			Length = 0;
			tailBlock = headBlock;
			tail = headBlock->Data - 1;

			// TODO: Would be interesting to re-sort blocks by size descending here
		}

		public void Push(T* value)
		{
			if (tail - tailBlock->Data == tailBlock->Capacity - 1)
			{
				tailBlock = tailBlock->NextBlock;
				tail = tailBlock->Data;
			}
			else
				++tail;

			*tail = value;
			Length++;
		}

		public int GetRequiredAllocationSize(int count, out PointerBlock<T>* parentBlock)
		{
			parentBlock = tailBlock;

			// Amount in current block
			int spaceLeftInCurrentBlock = tailBlock->Capacity - (int) (tail + 1 - tailBlock->Data);
			count -= spaceLeftInCurrentBlock;
			if (count <= 0) return 0;

			// Amount in subsequent blocks
			while (parentBlock->NextBlock != null)
			{
				parentBlock = parentBlock->NextBlock;
				count -= parentBlock->Capacity;
				if (count <= 0) return 0;
			}

			return max(count, parentBlock->Capacity * 2);
		}

		public T* Pop()
		{
			T* previousTail = *tail;

			if (tail == tailBlock->Data && Length > 1)
			{
				tailBlock = tailBlock->PreviousBlock;
				tail = tailBlock->Data + (tailBlock->Capacity - 1);
			}
			else
				--tail;

			Length--;

			return previousTail;
		}
	}

	public unsafe struct PointerBlock<T> where T : unmanaged
	{
		public readonly int Capacity;
		public readonly T** Data;
		public readonly PointerBlock<T>* PreviousBlock;

		public PointerBlock<T>* NextBlock;

		public PointerBlock(T** data, int capacity, PointerBlock<T>* previousBlock = null) : this()
		{
			Capacity = capacity;
			Data = data;
			PreviousBlock = previousBlock;
		}
	}
}