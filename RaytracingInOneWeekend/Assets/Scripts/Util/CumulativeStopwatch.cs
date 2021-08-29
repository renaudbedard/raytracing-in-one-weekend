using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Util
{
	readonly struct CumulativeStopwatch : IDisposable
	{
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
				UnityEngine.Debug.Log($"{kvp.Key} : {kvp.Value.Item2.Elapsed} total, {kvp.Value.Item2.Elapsed.TotalMilliseconds / kvp.Value.Item1:f2} ms average ({kvp.Value.Item1} hits)");

			stopwatches.Clear();
		}
	}
}