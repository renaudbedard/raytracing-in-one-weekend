using System;
using System.Diagnostics;

namespace Util
{
	readonly struct ScopedStopwatch : IDisposable
	{
		readonly string name;
		readonly Stopwatch stopwatch;

		public ScopedStopwatch(string name)
		{
			this.name = name;
			stopwatch = Stopwatch.StartNew();
		}

		public void Dispose()
		{
			stopwatch.Stop();
			UnityEngine.Debug.Log($"{name} : {stopwatch.Elapsed}");
		}
	}
}