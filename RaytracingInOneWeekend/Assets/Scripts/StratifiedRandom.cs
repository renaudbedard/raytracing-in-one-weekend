using Unity.Mathematics;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	struct StratifiedRandom
	{
		readonly int divisions;
		readonly float regionSize;
		int index;
		Random rng;

		public StratifiedRandom(uint seed, int start, int period)
		{
			divisions = (int) floor(sqrt(period));
			regionSize = 1.0f / divisions;
			rng = new Random(seed);
			index = start;
		}

		public float2 NextFloat2()
		{
			int2 cell = int2(index / divisions % divisions,index % divisions);
			index++;
			float2 from = float2(cell) * regionSize, to = from + regionSize;
			//Debug.Log($"From ({from.x}, {from.y}) to ({to.x}, {to.y}) for index {index}");
			return rng.NextFloat2(from, to);
		}
	}
}