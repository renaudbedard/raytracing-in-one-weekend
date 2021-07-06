using UnityEngine.Assertions;

namespace RaytracerInOneWeekend
{
	public unsafe struct PointerBlockChain<T> where T : unmanaged
	{
		public int Length { get; private set; }

		readonly PointerBlock<T>* headBlock;

		PointerBlock<T>* tailBlock;
		T** tail;

		public PointerBlock<T>* TailBlock => tailBlock;

		public PointerBlockChain(PointerBlock<T>* firstBlock) : this()
		{
			Assert.IsTrue(firstBlock != null, "First block must be allocated");
			headBlock = tailBlock = firstBlock;
			tail = firstBlock->Data - 1; // This will never be accessed
		}

		public void Clear()
		{
			Length = 0;
			tailBlock = headBlock;
			tail = headBlock->Data - 1;
		}

		public void Push(T* value)
		{
			if (tail == tailBlock->Data + (tailBlock->Capacity - 1))
			{
				Assert.IsFalse(tailBlock->NextBlock == null, "No space in tail block, use TryPush instead");
				tailBlock = tailBlock->NextBlock;
				tail = tailBlock->Data;
			}
			else
				++tail;

			*tail = value;
			Length++;
		}

		public bool TryPush(T* value)
		{
			if (tail == tailBlock->Data + (tailBlock->Capacity - 1))
			{
				if (tailBlock->NextBlock == null)
					return false;

				tailBlock = tailBlock->NextBlock;
				tail = tailBlock->Data;
			}
			else
				++tail;

			*tail = value;
			Length++;
			return true;
		}

		public T* Pop()
		{
			Assert.IsTrue(Length > 0, "Nothing to pop!");

			T* previousTail = *tail;

			if (tail == tailBlock->Data && Length > 1)
			{
				tailBlock = tailBlock->PreviousBlock;
				Assert.IsFalse(tailBlock == null, "No previous block found");
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
			Assert.IsTrue(capacity >= 1, "Capacity must be at least 1");
			Capacity = capacity;
			Data = data;
			PreviousBlock = previousBlock;
		}
	}
}