using System;
using System.Threading;
using UnityEditor;

namespace Util
{
	[InitializeOnLoad]
	public class SyncContextUtil
	{
		static readonly SynchronizationContext MainThreadSynchronizationContext;

		static SyncContextUtil()
		{
			MainThreadSynchronizationContext = SynchronizationContext.Current;
		}

		public static void EnsureOnMainThread(Action a)
		{
			if (SynchronizationContext.Current == MainThreadSynchronizationContext)
				a();
			else
				MainThreadSynchronizationContext.Post(_ => a(), null);
		}
	}
}