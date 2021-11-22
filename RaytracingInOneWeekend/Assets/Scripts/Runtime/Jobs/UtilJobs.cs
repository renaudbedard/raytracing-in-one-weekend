using System.Diagnostics;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Util;

namespace Runtime.Jobs
{
	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct ClearBufferJob<T> : IJob where T : unmanaged
	{
		[ReadOnly] public NativeReference<bool> CancellationToken;

		[WriteOnly] public NativeArray<T> Buffer;

		public void Execute()
		{
			if (CancellationToken.Value)
				return;

			Buffer.ZeroMemory();
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct CopyFloatBufferJob : IJob
	{
		[ReadOnly] public NativeReference<bool> CancellationToken;

		[ReadOnly] public NativeArray<float> Input;
		[WriteOnly] public NativeArray<float> Output;

		public void Execute()
		{
			if (CancellationToken.Value)
				return;

			NativeArray<float>.Copy(Input, Output);
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct CopyFloat3BufferJob : IJob
	{
		[ReadOnly] public NativeReference<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> Input;
		[WriteOnly] public NativeArray<float3> Output;

		public void Execute()
		{
			if (CancellationToken.Value)
				return;

			NativeArray<float3>.Copy(Input, Output);
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct CopyFloat4BufferJob : IJob
	{
		[ReadOnly] public NativeReference<bool> CancellationToken;

		[ReadOnly] public NativeArray<float4> Input;
		[WriteOnly] public NativeArray<float4> Output;

		public void Execute()
		{
			if (CancellationToken.Value)
				return;

			NativeArray<float4>.Copy(Input, Output);
		}
	}

	struct RecordTimeJob : IJob
	{
		[ReadOnly] public int Index;
		[WriteOnly] public NativeArray<long> Buffer;

		public void Execute()
		{
			Buffer[Index] = Stopwatch.GetTimestamp();
		}
	}
}