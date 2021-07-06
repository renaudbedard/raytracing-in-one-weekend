using System;
using System.Diagnostics;
using UnityEngine.Assertions;
using Debug = UnityEngine.Debug;

namespace RaytracerInOneWeekend
{
	public unsafe struct PointerBlockChain<T> where T : unmanaged
	{
		public int Length { get; private set; }

		readonly PointerBlock<T>* headBlock;

		PointerBlock<T>* tailBlock;
		T** tail;

		int blockIndex;
		readonly string chainId;

		public PointerBlock<T>* TailBlock => tailBlock;

		public PointerBlockChain(PointerBlock<T>* firstBlock) : this()
		{
			chainId = Guid.NewGuid().ToString().Substring(0, 4);

			Assert.IsTrue(firstBlock != null, "First block must be allocated");
			headBlock = firstBlock;
			Clear();
		}

		public void Clear()
		{
			Length = 0;
			tailBlock = headBlock;
			tail = headBlock->Data - 1;
			blockIndex = 0;
		}

		public void Push(T* value)
		{
			if (tail - tailBlock->Data == tailBlock->Capacity - 1)
			{
				Assert.IsFalse(tailBlock->NextBlock == null, "No space in tail block, use TryPush instead");
				Assert.IsTrue(tailBlock == tailBlock->NextBlock->PreviousBlock, "Block chain is invalid");
				tailBlock = tailBlock->NextBlock;
				tail = tailBlock->Data;
				blockIndex++;
				Debug.Log($"[{chainId}] [Push] Moved to next block (from {blockIndex - 1} to {blockIndex})");
			}
			else
				++tail;

			*tail = value;
			Length++;

			Debug.Log($"[{chainId}] [Push] PUSHED : Tail now {tail - tailBlock->Data + 1}/{tailBlock->Capacity} in block {blockIndex}, length = {Length}");
		}

		public bool TryPush(T* value)
		{
			if (tail - tailBlock->Data == tailBlock->Capacity - 1)
			{
				if (tailBlock->NextBlock == null)
				{
					Debug.Log($"[{chainId}] [TryPush] No space in block {blockIndex} (capacity = {tailBlock->Capacity}), exiting");
					return false;
				}

				tailBlock = tailBlock->NextBlock;
				tail = tailBlock->Data;
				blockIndex++;
				Debug.Log($"[{chainId}] [TryPush] Moved to next block (from {blockIndex - 1} to {blockIndex})");
			}
			else
				++tail;

			try
			{
				*tail = value;
			}
			catch
			{
				Assert.IsTrue(false, "Block chain is invalid (can't write to block tail)");
			}

			Length++;

			Debug.Log($"[{chainId}] [TryPush] PUSHED : Tail now {tail - tailBlock->Data + 1}/{tailBlock->Capacity} in block {blockIndex}, length = {Length}");
			return true;
		}

		public T* Pop()
		{
			Assert.IsTrue(Length > 0, "Nothing to pop!");

			T* previousTail = *tail;

			if (tail == tailBlock->Data && Length > 1)
			{
				Assert.IsTrue(tailBlock->PreviousBlock != null, "No previous block");
				try
				{
					Assert.IsTrue(tailBlock->PreviousBlock->NextBlock == tailBlock, "Block chain is invalid (previous block does not link back to tail block)");
				}
				catch
				{
					Assert.IsTrue(false, "Block chain is invalid (can't read from tail block)");
				}
				tailBlock = tailBlock->PreviousBlock;
				tail = tailBlock->Data + (tailBlock->Capacity - 1);
				blockIndex--;
				Debug.Log($"[{chainId}] [Pop] Moved to previous block (from {blockIndex + 1} to {blockIndex})");
			}
			else
				--tail;

			Length--;

			Debug.Log($"[{chainId}] [Pop] POPPED : Tail now {tail - tailBlock->Data + 1}/{tailBlock->Capacity} in block {blockIndex}, length = {Length}");
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