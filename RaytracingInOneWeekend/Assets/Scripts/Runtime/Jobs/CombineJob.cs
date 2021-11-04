using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime.Jobs
{
	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct CombineJob : IJobParallelFor
	{
		static readonly float3 NoSamplesColor = new(1, 0, 1);
		static readonly float3 NaNColor = new(0, 1, 1);

		public bool DebugMode;
		public bool LdrAlbedo;

		[ReadOnly] public NativeReference<bool> CancellationToken;

		[ReadOnly] public NativeArray<float4> InputColor;
		[ReadOnly] public NativeArray<float3> InputNormal;
		[ReadOnly] public NativeArray<float3> InputAlbedo;
		[ReadOnly] public int2 Size;

		[WriteOnly] public NativeArray<float3> OutputColor;
		[WriteOnly] public NativeArray<float3> OutputNormal;
		[WriteOnly] public NativeArray<float3> OutputAlbedo;

		public void Execute(int index)
		{
			if (CancellationToken.Value)
				return;

			float4 inputColor = InputColor[index];
			var realSampleCount = (int) inputColor.w;

			float3 finalColor;
			if (!DebugMode)
			{
				if (realSampleCount == 0)
				{
					int tentativeIndex = index;

					// look-around (for interlaced buffer)
					while (realSampleCount == 0 && (tentativeIndex -= Size.x) >= 0)
					{
						inputColor = InputColor[tentativeIndex];
						realSampleCount = (int) inputColor.w;
					}
				}

				if (realSampleCount == 0) finalColor = 0;
				else if (any(isnan(inputColor))) finalColor = 0;
				else finalColor = inputColor.xyz / realSampleCount;
			}
			else
			{
				if (realSampleCount == 0) finalColor = NoSamplesColor;
				else if (any(isnan(inputColor))) finalColor = NaNColor;
				else finalColor = inputColor.xyz / realSampleCount;
			}

			float3 finalAlbedo = InputAlbedo[index] / max(realSampleCount, 1);

			if (LdrAlbedo)
				finalAlbedo = min(finalAlbedo, 1);

			OutputColor[index] = finalColor;
			OutputNormal[index] = normalizesafe(InputNormal[index] / max(realSampleCount, 1));
			OutputAlbedo[index] = finalAlbedo;
		}
	}
}