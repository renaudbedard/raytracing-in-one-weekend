using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime
{
	static class R2
	{
		public static float2 Next(uint n)
		{
			// from : http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
			const float g = 1.32471795724474602596f;
			const float a1 = 1.0f / g;
			const float a2 = 1.0f / (g * g);

			return float2((0.5f + a1 * n) % 1, (0.5f + a2 * n) % 1);
		}
	}
}