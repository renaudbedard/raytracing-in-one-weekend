using System;

#if PROFILING
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
#endif

namespace Util
{
	readonly struct CumulativeStopwatch : IDisposable
	{
#if PROFILING
		static readonly Dictionary<string, (int, Stopwatch)> stopwatches = new Dictionary<string, (int, Stopwatch)>();

		readonly string name;
		readonly (int HitCount, Stopwatch Stopwatch) state;

		public CumulativeStopwatch(string name)
		{
			this.name = name;

			if (stopwatches.TryGetValue(name, out state))
				state.Stopwatch.Start();
			else
				state = (0, Stopwatch.StartNew());
		}

		public void Dispose()
		{
			state.Stopwatch.Stop();
			stopwatches[name] = (state.HitCount + 1, state.Stopwatch);
		}

		public static void Log()
		{
			foreach (KeyValuePair<string, (int, Stopwatch)> kvp in stopwatches.OrderByDescending(x => x.Value.Item2.Elapsed))
				UnityEngine.Debug.Log($"{kvp.Key} : {kvp.Value.Item2.Elapsed} total, {kvp.Value.Item2.Elapsed.TotalMilliseconds / kvp.Value.Item1:f3} ms average ({kvp.Value.Item1} hits)");

			stopwatches.Clear();
		}
#else
		public CumulativeStopwatch(string _) { }
		public void Dispose() { }
#endif
	}
}