using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	enum ImportanceSamplingMode
	{
		None,
		LightsOnly,
		Mixture
	}

	struct ImportanceSampler
	{
		[ReadOnly] public NativeArray<Entity> TargetEntities;
		public ImportanceSamplingMode Mode;

		public unsafe void Sample(Ray materialScatterRay, float materialScatteringPdfValue, ref Random rng,
			out Ray scatterRay, out float pdfValue)
		{
			int totalOptions = TargetEntities.Length + (Mode == ImportanceSamplingMode.Mixture ? 1 : 0);
			int chosenOption = rng.NextInt(0, totalOptions);
			if (chosenOption == TargetEntities.Length)
				scatterRay = materialScatterRay;
			else
			{
				Entity* chosenEntity = (Entity*) TargetEntities.GetUnsafeReadOnlyPtr() + chosenOption;
				float3 pointOnEntity = chosenEntity->RandomPoint(materialScatterRay.Time, ref rng);
				scatterRay = new Ray(materialScatterRay.Origin,
					normalize(pointOnEntity - materialScatterRay.Origin));
			}

			pdfValue = 0;
			if (Mode == ImportanceSamplingMode.Mixture) pdfValue += materialScatteringPdfValue;

			var basePointer = (Entity*) TargetEntities.GetUnsafeReadOnlyPtr();
			for (int i = 0; i < TargetEntities.Length; i++)
				pdfValue += (basePointer + i)->PdfValue(scatterRay, ref rng);
			pdfValue /= totalOptions;
		}
	}
}