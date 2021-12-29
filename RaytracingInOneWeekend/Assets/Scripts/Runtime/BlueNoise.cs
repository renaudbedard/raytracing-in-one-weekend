using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Runtime
{
	unsafe struct BlueNoise
	{
		[NativeDisableUnsafePtrRestriction] readonly half4* noiseData;
		readonly uint rowStride;
		readonly uint seed;

		PerPixelNoise<half4> perPixelNoise;

		public BlueNoise(uint seed, half4* noiseData, uint rowStride) : this()
		{
			this.seed = seed;
			this.noiseData = noiseData;
			this.rowStride = rowStride;
		}

		public uint2 Coordinates
		{
			set => perPixelNoise = new PerPixelNoise<half4>(seed, value, noiseData, rowStride);
		}

		public float NextFloat() => perPixelNoise.Next().x;

		public float2 NextFloat2() => perPixelNoise.Next().xy;
	}
}