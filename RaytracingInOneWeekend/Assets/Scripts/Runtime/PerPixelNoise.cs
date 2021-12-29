using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime
{
	unsafe struct PerPixelNoise<T> where T : unmanaged
	{
		[NativeDisableUnsafePtrRestriction] readonly T* noiseData;
		readonly uint2 coordinates;
		uint2 offset;
		readonly uint rowStride;
		uint n;

		public PerPixelNoise(uint seed, uint2 coordinates, T* noiseData, uint rowStride) : this()
		{
			this.coordinates = coordinates;
			this.noiseData = noiseData;
			this.rowStride = rowStride;

			n = (byte) (seed % 255);
			Advance();
		}

		public T Next()
		{
			uint2 wrappedCoords = (coordinates + offset) % rowStride;
			T* pPixel = noiseData + wrappedCoords.y * rowStride + wrappedCoords.x;
			Advance();
			return *pPixel;
		}

		void Advance()
		{
			offset += (uint2) floor(R2.Next(n++) * rowStride);
		}
	}
}