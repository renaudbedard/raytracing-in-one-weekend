using System;
using System.Collections.Generic;
using UnityEngine;
using Object = UnityEngine.Object;

#if ODIN_INSPECTOR
using Sirenix.Utilities;
#endif

namespace Util
{
	public class Pool<T> : IDisposable where T : new()
	{
		readonly Action<T> cleanupMethod;
		readonly Func<T> factoryMethod;
		readonly Func<T, string> itemNameMethod;
		readonly Dictionary<T, int> takenItems;

		int freeIndex;
		T[] items = Array.Empty<T>();

		public Pool(Func<T> factoryMethod = null, Action<T> cleanupMethod = null, Func<T, string> itemNameMethod = null, IEqualityComparer<T> equalityComparer = null)
		{
			takenItems = new Dictionary<T, int>(equalityComparer ?? EqualityComparer<T>.Default);

			this.factoryMethod = factoryMethod;
			this.cleanupMethod = cleanupMethod;
			this.itemNameMethod = itemNameMethod ?? (x => x.ToString());
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
					throw new InvalidOperationException($"Reducing capacity of {typeof(T).GetNiceName()} pool to {value} would make pool lose track of {freeIndex - value} taken items");

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
				Debug.Log($"Grew pool of {typeof(T).GetNiceName()} from {oldCapacity} to {items.Length}");
			}

			T item = items[freeIndex];
			takenItems[item] = freeIndex;
#if TRACE_LOGGING
			Debug.Log($"Taking {typeof(T).GetNiceName()} {itemNameMethod(item)} at #{freeIndex}");
#endif
			++freeIndex;

			return item;
		}

		public void Return(T item)
		{
			if (!takenItems.TryGetValue(item, out int sourceIndex))
			{
				Debug.LogError($"{typeof(T).GetNiceName()} {itemNameMethod(item)} count not be found in TakenIndices ({TakenCount} taken)");
				return;
			}

			int destinationIndex = --freeIndex;
			cleanupMethod?.Invoke(item);
#if TRACE_LOGGING
			Debug.Log($"Returned {typeof(T).GetNiceName()} {itemNameMethod(item)} at #{sourceIndex}, swapping with #{destinationIndex}");
#endif
			takenItems.Remove(item);

			T destinationItem = items[destinationIndex];
			(items[destinationIndex], items[sourceIndex]) = (items[sourceIndex], destinationItem);
			takenItems[destinationItem] = sourceIndex;
		}

		public void ReturnAll()
		{
			if (cleanupMethod != null)
				for (var i = 0; i < freeIndex; i++)
					cleanupMethod(items[i]);

			freeIndex = 0;
			takenItems.Clear();
		}
	}

#if !ODIN_INSPECTOR
	static class TypeExtensions
	{
		public static string GetNiceName(this Type type) => type.Name;
	}
#endif
}