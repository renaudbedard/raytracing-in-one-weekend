using System;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	struct StratifiedRandom
	{
		enum CellSamplingMode
		{
			Random,
			Center,
			MinX,
			MaxX
		}

		readonly int2 divisions;
		readonly float2 regionSize;
		int index;
		Random rng;
		CellSamplingMode mode;

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
				case CellSamplingMode.MinX: return float2(from.x, (from.y + to.y) / 2);
				case CellSamplingMode.MaxX: return float2(to.x, (from.y + to.y) / 2);
			}
			throw new NotSupportedException("Unsupported sampling mode");
		}

		public float SolidAngle
		{
			get
			{
				CellSamplingMode lastMode = mode;
				--index; mode = CellSamplingMode.MinX;
				float3 minVector = this.OnUniformHemisphere(float3(0, 1, 0));
				--index; mode = CellSamplingMode.MaxX;
				float3 maxVector = this.OnUniformHemisphere(float3(0, 1, 0));
				mode = lastMode;
				return radians(Vector3.Angle(minVector, maxVector));
			}
		}
	}
}