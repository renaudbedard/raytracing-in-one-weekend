using UnityEngine.Assertions;
using Debug = UnityEngine.Debug;

namespace Util
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
			Debug.Log($"Pushing value in block of {tailBlock->Capacity} ({(int)tailBlock:x8}), currently at {tail - tailBlock->Data} ({(int)tail:x8} - {(int)tailBlock->Data:x8})");

			if (tail - tailBlock->Data == tailBlock->Capacity - 1)
			{
				Assert.IsFalse(tailBlock->NextBlock == null, "No space in tail block, use TryPush instead");
				Assert.IsTrue(tailBlock == tailBlock->NextBlock->PreviousBlock, "Block chain is invalid");
				tailBlock = tailBlock->NextBlock;
				tail = tailBlock->Data;
				Debug.Log($"Moved to next block (of {tailBlock->Capacity})");
			}
			else
			{
				++tail;
				Debug.Log($"Advanced tail in current block, now at {tail - tailBlock->Data}");
			}

			*tail = value;
			Length++;
		}

		public bool TryPush(T* value)
		{
			Debug.Log($"Trying to push value in block of {tailBlock->Capacity} ({(int)tailBlock:x8}), currently at {tail - tailBlock->Data} ({(int)tail:x8} - {(int)tailBlock->Data:x8})");

			if (tail - tailBlock->Data == tailBlock->Capacity - 1)
			{
				if (tailBlock->NextBlock == null)
				{
					Debug.Log("No more space in tail block and next block is null; failed");
					return false;
				}

				tailBlock = tailBlock->NextBlock;
				tail = tailBlock->Data;
				Debug.Log($"Moved to next block (of {tailBlock->Capacity})");
			}
			else
			{
				++tail;
				Debug.Log($"Advanced tail in current block, now at {tail - tailBlock->Data}");
			}

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
				Assert.IsTrue(tailBlock->PreviousBlock != null, "No previous block");
				Assert.IsTrue(tailBlock->PreviousBlock->NextBlock == tailBlock, "Block chain is invalid (previous block does not link back to tail block)");
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
			Assert.IsTrue(capacity >= 1, "Capacity must be at least 1");
			Capacity = capacity;
			Data = data;
			PreviousBlock = previousBlock;
		}
	}
}