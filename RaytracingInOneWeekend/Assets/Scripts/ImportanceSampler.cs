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

		public void Sample(Ray materialScatterRay, float materialScatteringPdfValue, ref Random rng,
			out Ray scatterRay, out float pdfValue)
		{
			int totalOptions = TargetEntities.Length + Mode == ImportanceSamplingMode.Mixture ? 1 : 0;
			int chosenOption = rng.NextInt(0, totalOptions);
			if (chosenOption == TargetEntities.Length)
			{
				scatterRay = materialScatterRay;
				pdfValue = materialScatteringPdfValue;
			}
			else unsafe
			{
				Entity* chosenEntity = (Entity*)TargetEntities.GetUnsafeReadOnlyPtr() + chosenOption;
				float3 pointOnEntity = chosenEntity->RandomPoint(materialScatterRay.Time, ref rng);
				scatterRay = new Ray(materialScatterRay.Origin, normalize(pointOnEntity - materialScatterRay.Origin));
				pdfValue = chosenEntity->PdfValue(scatterRay, ref rng);
			}
		}
	}
}