using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Runtime
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

		public unsafe void Sample(Ray materialScatterRay, float3 outgoingLightDirection, HitRecord rec, Material* material, ref RandomSource rng,
			out Ray scatterRay, out float pdfValue, out Entity* targetEntityId)
		{
			int totalOptions = TargetEntities.Length + (Mode == ImportanceSamplingMode.Mixture ? 1 : 0);
			int chosenOption = rng.NextInt(0, totalOptions);
			if (chosenOption == TargetEntities.Length)
			{
				scatterRay = materialScatterRay;
				targetEntityId = null;
			}
			else
			{
				Entity* chosenEntity = (Entity*) TargetEntities.GetUnsafeReadOnlyPtr() + chosenOption;
				float3 pointOnEntity = chosenEntity->RandomPoint(materialScatterRay.Time, ref rng);
				scatterRay = new Ray(materialScatterRay.Origin,
					normalize(pointOnEntity - materialScatterRay.Origin));
				targetEntityId = chosenEntity;
			}

			pdfValue = 0;
			if (Mode == ImportanceSamplingMode.Mixture)
				pdfValue += material->Pdf(scatterRay.Direction, outgoingLightDirection, rec.Normal);

			var basePointer = (Entity*) TargetEntities.GetUnsafeReadOnlyPtr();
			for (int i = 0; i < TargetEntities.Length; i++)
				pdfValue += (basePointer + i)->Pdf(scatterRay);
			pdfValue /= totalOptions;
		}
	}
}