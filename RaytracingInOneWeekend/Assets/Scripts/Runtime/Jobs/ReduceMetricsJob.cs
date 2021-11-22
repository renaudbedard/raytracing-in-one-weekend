using Unity;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime.Jobs
{
	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct ReduceMetricsJob : IJob
	{
		[ReadOnly] public NativeArray<Diagnostics> Diagnostics;
		[ReadOnly] public NativeArray<float4> AccumulatedColor;
		[ReadOnly] public NativeArray<float> AccumulatedSampleCountWeight;

		[WriteOnly] public NativeReference<int> TotalRayCount;
		[WriteOnly] public NativeReference<int> TotalSamples;
		[WriteOnly] public NativeReference<float2> SampleCountWeightExtrema;
		[WriteOnly] public NativeReference<int2> SampleCountExtrema;

		public void Execute()
		{
			int totalRayCount = 0;
			float minSampleCountWeight = INFINITY, maxSampleCountWeight = -INFINITY;
			float minSamples = INFINITY, maxSamples = -INFINITY;
			int totalSamples = 0;

			for (int i = 0; i < Diagnostics.Length; i++)
			{
				totalRayCount += (int) Diagnostics[i].RayCount;
				var sampleCount = (int) AccumulatedColor[i].w;
				totalSamples += sampleCount;
				float sampleCountWeight = AccumulatedSampleCountWeight[i] / sampleCount;
				minSampleCountWeight = min(minSampleCountWeight, sampleCountWeight);
				maxSampleCountWeight = max(maxSampleCountWeight, sampleCountWeight);
				minSamples = min(minSamples, sampleCount);
				maxSamples = max(maxSamples, sampleCount);
			}

			TotalRayCount.Value = totalRayCount;
			TotalSamples.Value = totalSamples;
			SampleCountWeightExtrema.Value = float2(minSampleCountWeight, maxSampleCountWeight);
			SampleCountExtrema.Value = int2((int) minSamples, (int) maxSamples);
		}
	}
}