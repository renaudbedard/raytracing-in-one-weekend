using Unity.Collections;
using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
	struct ImportanceSampler
	{
		[ReadOnly] public NativeArray<Entity> TargetEntities;

		public (Ray scatterRay, float pdfValue) Sample(Ray materialScatterRay, float materialScatteringPdfValue, ref Random rng)
		{
			int totalOptions = TargetEntities.Length + 1;
			int chosenOption = rng.NextInt(0, totalOptions);
			if (chosenOption == TargetEntities.Length)
				return (materialScatterRay, materialScatteringPdfValue);
			else
			{
				float3 pointOnEntity = TargetEntities[chosenOption].RandomPoint(materialScatterRay.Time, ref rng);
				var scatterRay = new Ray(materialScatterRay.Origin, pointOnEntity - materialScatterRay.Origin);
				float pdfValue = TargetEntities[chosenOption].PdfValue(scatterRay, ref rng);
				return (scatterRay, pdfValue);
			}
		}
	}
}