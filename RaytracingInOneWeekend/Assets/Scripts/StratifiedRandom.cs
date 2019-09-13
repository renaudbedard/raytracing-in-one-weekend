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

		public StratifiedRandom(uint seed, int start, int period)
		{
			float d = sqrt(period);
			divisions = int2(ceil(float2(d / 2.0f, period / (d / 2.0f))));
			regionSize = float2(1.0f) / divisions;
			rng = new Random(seed);
			index = start;
			mode = CellSamplingMode.Center;
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