using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	struct StratifiedRandom
	{
		readonly int2 divisions;
		readonly float2 regionSize;
		int index;
		Random rng;

		public StratifiedRandom(uint seed, int start, int period)
		{
			float d = sqrt(period);
			divisions = int2(ceil(float2(d / 2.0f, period / (d / 2.0f))));
			regionSize = float2(1.0f) / divisions;
			rng = new Random(seed);
			index = start;
		}

		public float2 NextFloat2()
		{
			int2 cell = int2(index / divisions.y % divisions.x,index % divisions.y);
			index++;
			float2 from = float2(cell) * regionSize, to = from + regionSize;
			return rng.NextFloat2(from, to);
		}
	}
}