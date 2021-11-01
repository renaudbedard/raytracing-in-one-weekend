using System;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Util
{
	public class Pool<T> : IDisposable where T : new()
	{
		readonly Action<T> cleanupMethod;
		readonly Func<T> factoryMethod;
		readonly string itemName;
		int freeIndex;

		T[] items = Array.Empty<T>();

		public Pool(Func<T> factoryMethod = null, Action<T> cleanupMethod = null, string itemNameOverride = null)
		{
			itemName = itemNameOverride ?? typeof(T).Name;

			this.factoryMethod = factoryMethod;
			this.cleanupMethod = cleanupMethod;
		}

		public void Reset()
		{
			ReturnAll();
			int oldCapacity = Capacity;
			Capacity = 0;
			Capacity = oldCapacity;
		}

		public int TakenCount => freeIndex;
		public int FreeCount => Capacity - TakenCount;

		public int Capacity
		{
			get => items.Length;
			set
			{
				if (freeIndex > value)
					throw new InvalidOperationException(
						$"Reducing capacity of {itemName} pool to {value} would make pool lose track of {freeIndex - value} taken items");

				int oldCapacity = items.Length;

				for (int i = value; i < oldCapacity; i++)
				{
					T item = items[i];

					switch (item)
					{
						case IDisposable disposable: disposable.Dispose(); break;
						case Object o: Object.Destroy(o); break;
					}
				}

				Array.Resize(ref items, value);

				for (int i = oldCapacity; i < items.Length; i++)
					items[i] = factoryMethod != null ? factoryMethod() : new T();
			}
		}

		public void Dispose()
		{
			ReturnAll();
			Capacity = 0;
		}

		public T Take()
		{
			if (freeIndex == items.Length)
			{
				int oldCapacity = items.Length;
				Capacity++;
				Debug.Log($"Grew pool of {itemName} from {oldCapacity} to {items.Length}");
			}

			return items[freeIndex++];
		}

		public void Return(T item)
		{
			freeIndex--;

			cleanupMethod?.Invoke(item);

			int destinationIndex = freeIndex;
			int sourceIndex = Array.IndexOf(items, item, 0, freeIndex + 1);

			(items[destinationIndex], items[sourceIndex]) = (items[sourceIndex], items[destinationIndex]);
		}

		public void ReturnAll()
		{
			if (cleanupMethod != null)
				for (var i = 0; i < freeIndex; i++)
					cleanupMethod(items[i]);

			freeIndex = 0;
		}
	}
}