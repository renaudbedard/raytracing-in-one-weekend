using Unity;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace Runtime.Jobs
{
	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct ReduceRayCountJob : IJob
	{
		[ReadOnly] public NativeArray<Diagnostics> Diagnostics;
		[WriteOnly] public NativeReference<int> TotalRayCount;

		public void Execute()
		{
			float totalRayCount = 0;

			for (int i = 0; i < Diagnostics.Length; i++)
				totalRayCount += Diagnostics[i].RayCount;

			TotalRayCount.Value = (int) totalRayCount;
		}
	}
}