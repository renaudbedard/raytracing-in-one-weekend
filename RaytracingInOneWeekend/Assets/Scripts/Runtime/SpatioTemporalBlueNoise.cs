using Unity.Collections.LowLevel.Unsafe;
using Util;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime
{
	unsafe struct SpatioTemporalBlueNoise
	{
		[NativeDisableUnsafePtrRestriction] readonly byte* scalarData;
		[NativeDisableUnsafePtrRestriction] readonly RGB24* vector2Data;
		[NativeDisableUnsafePtrRestriction] readonly RGBA32* cosineUnitVector3Data;
		[NativeDisableUnsafePtrRestriction] readonly RGB24* unitVector2Data;
		[NativeDisableUnsafePtrRestriction] readonly RGB24* unitVector3Data;

		readonly uint rowStride;
		readonly uint seed;

		PerPixelNoise<byte> perPixelScalar;
		PerPixelNoise<RGB24> perPixelVector2;
		PerPixelNoise<RGBA32> perPixelCosineUnitVector3;
		PerPixelNoise<RGB24> perPixelUnitVector2;
		PerPixelNoise<RGB24> perPixelUnitVector3;

		public SpatioTemporalBlueNoise(uint seed,
			byte* scalarData,
			RGB24* vector2Data,
			RGBA32* cosineUnitVector3Data,
			RGB24* unitVector2Data,
			RGB24* unitVector3Data,
			uint rowStride) : this()
		{
			this.rowStride = rowStride;
			this.scalarData = scalarData;
			this.vector2Data = vector2Data;
			this.cosineUnitVector3Data = cosineUnitVector3Data;
			this.unitVector2Data = unitVector2Data;
			this.unitVector3Data = unitVector3Data;
			this.seed = seed;
		}

		public uint2 Coordinates
		{
			set
			{
				perPixelScalar = new PerPixelNoise<byte>(seed, value, scalarData, rowStride);
				perPixelVector2 = new PerPixelNoise<RGB24>(seed, value, vector2Data, rowStride);
				perPixelCosineUnitVector3 = new PerPixelNoise<RGBA32>(seed, value, cosineUnitVector3Data, rowStride);
				perPixelUnitVector2 = new PerPixelNoise<RGB24>(seed, value, unitVector2Data, rowStride);
				perPixelUnitVector3 = new PerPixelNoise<RGB24>(seed, value, unitVector3Data, rowStride);
			}
		}

		public float NextFloat() => perPixelScalar.Next() / 256.0f;

		public float2 NextFloat2()
		{
			RGB24 value = perPixelVector2.Next();
			return float2(value.r, value.g) / 256.0f;
		}

		public float3 NextCosineUnitVector3()
		{
			RGBA32 value = perPixelCosineUnitVector3.Next();
			return float3(value.r, value.b, value.g) / 256.0f * 2 - 1;
		}

		public float2 NextUnitVector2()
		{
			RGB24 value = perPixelUnitVector2.Next();
			return float2(value.r, value.g) / 256.0f * 2 - 1;
		}

		public float3 NextUnitVector3()
		{
			RGB24 value = perPixelUnitVector3.Next();
			return float3(value.r, value.g, value.b) / 256.0f * 2 - 1;
		}
	}
}