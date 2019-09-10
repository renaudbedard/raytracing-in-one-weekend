using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	struct StratifiedRandom
	{
		private int divisions;
		private int current;
		private Random rng;

		public StratifiedRandom(uint seed, int sampleCount)
		{
			divisions = (int) floor(sqrt(sampleCount));
			rng = new Random(seed);
			current = 0;
		}

		public float2 Sample()
		{
			int2 cell = int2(current / divisions % divisions,current % divisions);
			// TODO
			rng.NextFloat2();
			current++;
			return default;
		}
	}
}