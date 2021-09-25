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

	unsafe struct ImportanceSampler
	{
		[ReadOnly] public UnsafePtrList<Entity> TargetEntityPointers;

		public ImportanceSamplingMode Mode;

		public void Sample(Ray materialScatterRay, float3 outgoingLightDirection, HitRecord rec, Material* material, ref RandomSource rng,
			out Ray scatterRay, out float pdfValue, out void* targetEntityContent)
		{
			int totalOptions = TargetEntityPointers.Length + (Mode == ImportanceSamplingMode.Mixture ? 1 : 0);
			int chosenOption = rng.NextInt(0, totalOptions);
			if (chosenOption == TargetEntityPointers.Length)
			{
				scatterRay = materialScatterRay;
				targetEntityContent = null;
			}
			else
			{
				Entity* chosenEntity = TargetEntityPointers[chosenOption];
				float3 pointOnEntity = chosenEntity->RandomPoint(materialScatterRay.Time, ref rng);
				scatterRay = new Ray(materialScatterRay.Origin, normalize(pointOnEntity - materialScatterRay.Origin));
				targetEntityContent = chosenEntity->Content;
			}

			pdfValue = 0;
			if (Mode == ImportanceSamplingMode.Mixture)
				pdfValue += material->Pdf(scatterRay.Direction, outgoingLightDirection, rec.Normal);

			for (int i = 0; i < TargetEntityPointers.Length; ++i)
				pdfValue += TargetEntityPointers[i]->Pdf(scatterRay);
			pdfValue /= totalOptions;
		}
	}
}