using System;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	struct StratifiedRandom
	{
		public enum CellSamplingMode
		{
			Random,
			Center,
			ZeroZero,
			ZeroOne,
			OneZero,
			OneOne
		}

		readonly int2 divisions;
		readonly float2 regionSize;
		int index;
		Random rng;
		CellSamplingMode mode;

		public int Index
		{
			get => index;
			set => index = value;
		}

		public CellSamplingMode Mode
		{
			get => mode;
			set => mode = value;
		}

		public int2 Divisions => divisions;

		public StratifiedRandom(uint seed, int start, int period)
		{
			// https://www.wolframalpha.com/input/?i=n+%3D+a+*+b%2C+b+%3D+2+*+a%2C+solve+for+a%2Cb
			float sqrtPeriod = sqrt(period);
			float sqrt2 = sqrt(2.0f);
			divisions = int2(ceil(float2(sqrtPeriod / sqrt2, sqrt2 * sqrtPeriod)));
			regionSize = float2(1.0f) / divisions;
			rng = new Random(seed);
			index = start;
			mode = CellSamplingMode.Random;
		}

		public float2 NextFloat2()
		{
			int2 cell = int2(index / divisions.y % divisions.x,index % divisions.y);
			index++;
			float2 from = float2(cell) * regionSize, to = from + regionSize;
			switch (mode)
			{
				case CellSamplingMode.Random: return rng.NextFloat2(from, to);
				case CellSamplingMode.Center: return (from + to) / 2;
				case CellSamplingMode.ZeroZero: return float2(from.x, from.y);
				case CellSamplingMode.ZeroOne: return float2(from.x, to.y);
				case CellSamplingMode.OneZero: return float2(to.x, from.y);
				case CellSamplingMode.OneOne: return float2(to.x, to.y);
			}
			throw new NotSupportedException("Unsupported sampling mode");
		}
	}
}