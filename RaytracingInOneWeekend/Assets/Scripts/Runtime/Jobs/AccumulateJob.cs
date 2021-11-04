using System.Diagnostics;
using Unity;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Util;
using Debug = UnityEngine.Debug;
using Random = Unity.Mathematics.Random;
using static Unity.Mathematics.math;

namespace Runtime.Jobs
{
	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	unsafe struct AccumulateJob : IJobParallelFor
	{
		const int BvhNodeStackAllocSize = 128;
		const int EntityStackAllocSize = 64;
		const int HitRecordStackAllocSize = 32;

		[ReadOnly] public NativeReference<bool> CancellationToken;

		[ReadOnly] public float2 Size;
		[ReadOnly] public int SliceOffset;
		[ReadOnly] public int SliceDivider;
		[ReadOnly] public uint Seed;
		[ReadOnly] public View View;
		[ReadOnly] public Environment Environment;
		[ReadOnly] public uint SampleCount;
		[ReadOnly] public int TraceDepth;
		[ReadOnly] public bool SubPixelJitter;
		[ReadOnly] [NativeDisableUnsafePtrRestriction] public BvhNode* BvhRoot;
		[ReadOnly] public PerlinNoise PerlinNoise;
		[ReadOnly] public BlueNoise BlueNoise;
		[ReadOnly] public NoiseColor NoiseColor;
		[ReadOnly] public NativeArray<float4> InputColor;
		[ReadOnly] public NativeArray<float3> InputNormal;
		[ReadOnly] public NativeArray<float3> InputAlbedo;

		[WriteOnly] public NativeArray<float4> OutputColor;
		[WriteOnly] public NativeArray<float3> OutputNormal;
		[WriteOnly] public NativeArray<float3> OutputAlbedo;

		[WriteOnly] public NativeArray<Diagnostics> OutputDiagnostics;

#if PATH_DEBUGGING
		[ReadOnly] public int2 DebugCoordinates;
		[NativeDisableUnsafePtrRestriction] public DebugPath* DebugPaths;
#endif

		[SkipLocalsInit]
		public void Execute(int index)
		{
			if (CancellationToken.Value)
				return;

			int2 coordinates = int2(
				index % (int) Size.x, // column
				index / (int) Size.x  // row
			);

			if (coordinates.y % SliceDivider != SliceOffset)
				return;

			float4 lastColor = InputColor[index];
			float3 colorAcc = lastColor.xyz;
			float3 normalAcc = InputNormal[index];
			float3 albedoAcc = InputAlbedo[index];

			int sampleCount = (int) lastColor.w;

			PerPixelBlueNoise blueNoise = default;
			Random whiteNoise = default;
			switch (NoiseColor)
			{
				case NoiseColor.Blue:
					blueNoise = BlueNoise.GetPerPixelData((uint2) coordinates);
					break;
				case NoiseColor.White:
					// Big primes stolen from Unity's random class
					whiteNoise = new Random((Seed * 0x8C4CA03Fu) ^ (uint) (index * 0x7383ED49u));
					break;
			}
			var rng = new RandomSource(NoiseColor, whiteNoise, blueNoise);

#if PATH_DEBUGGING
			bool doDebugPaths = all(coordinates == DebugCoordinates);
			if (doDebugPaths)
				for (int i = 0; i < TraceDepth; i++)
					DebugPaths[i] = default;
#endif

			float3* emissionStack = stackalloc float3[TraceDepth];
			float3* attenuationStack = stackalloc float3[TraceDepth];

			BvhNode** bvhNodeBuffer = stackalloc BvhNode*[BvhNodeStackAllocSize];
			var nodeTraversalStack = new HybridPtrStack<BvhNode>(bvhNodeBuffer, BvhNodeStackAllocSize);

			Entity** entityBuffer = stackalloc Entity*[EntityStackAllocSize];
			var hitCandidateStack = new HybridPtrStack<Entity>(entityBuffer, EntityStackAllocSize);

			HitRecord* hitRecordBuffer = stackalloc HitRecord[HitRecordStackAllocSize];
			var hitRecordList = new HybridList<HitRecord>(hitRecordBuffer, HitRecordStackAllocSize);

			float3 fallbackAlbedo = default, fallbackNormal = default;
			Diagnostics diagnostics = default;

			for (int s = 0; s < SampleCount; s++)
			{
				float2 normalizedCoordinates = (coordinates + (SubPixelJitter ? rng.NextFloat2() : 0.5f)) / Size;
				Ray eyeRay = View.GetRay(normalizedCoordinates, ref rng);

				if (Sample(eyeRay, ref rng, emissionStack, attenuationStack, ref nodeTraversalStack, ref hitCandidateStack, ref hitRecordList,
#if PATH_DEBUGGING
					doDebugPaths && s == 0,
#endif
					out float3 sampleColor, out float3 sampleNormal, out float3 sampleAlbedo, ref diagnostics))
				{
					colorAcc += sampleColor;
					normalAcc += sampleNormal;
					albedoAcc += sampleAlbedo;

					sampleCount++;
				}

				if (s == 0)
				{
					fallbackNormal = sampleNormal;
					fallbackAlbedo = sampleAlbedo;
				}
			}

			OutputColor[index] = float4(colorAcc, sampleCount);
			OutputNormal[index] = sampleCount == 0 ? fallbackNormal : normalAcc;
			OutputAlbedo[index] = sampleCount == 0 ? fallbackAlbedo : albedoAcc;

			OutputDiagnostics[index] = diagnostics;
		}

		bool Sample(Ray eyeRay, ref RandomSource rng, float3* emissionStack, float3* attenuationStack,
			ref HybridPtrStack<BvhNode> nodeTraversalBuffer, ref HybridPtrStack<Entity> hitCandidateBuffer, ref HybridList<HitRecord> hitRecordBuffer,
#if PATH_DEBUGGING
			bool doDebugPaths,
#endif
			out float3 sampleColor, out float3 sampleNormal, out float3 sampleAlbedo, ref Diagnostics diagnostics)
		{
			float3* emissionCursor = emissionStack;
			float3* attenuationCursor = attenuationStack;

			int depth = 0;
			void* explicitSamplingTarget = null;
			bool firstNonSpecularHit = false;
			sampleColor = sampleNormal = sampleAlbedo = default;
			Material* currentProbabilisticVolumeMaterial = null;
			bool stopTracing = false;

			Ray ray = eyeRay;

			for (; depth < TraceDepth; depth++)
			{
				FindHitCandidates(ray, BvhRoot, ref nodeTraversalBuffer, ref hitCandidateBuffer
#if FULL_DIAGNOSTICS
					, ref diagnostics
#endif
					);

				FindHits(ray, ref hitCandidateBuffer, ref hitRecordBuffer);

				if (currentProbabilisticVolumeMaterial == null && explicitSamplingTarget == null)
				{
					DetermineVolumeContainment(ray, BvhRoot, ref nodeTraversalBuffer, ref hitCandidateBuffer, hitRecordBuffer,
#if FULL_DIAGNOSTICS
						ref diagnostics,
#endif
						out currentProbabilisticVolumeMaterial);
				}

				diagnostics.RayCount++;

				int hitIndex = 0;
				while (hitIndex < hitRecordBuffer.Length)
				{
					HitRecord rec = hitRecordBuffer[hitIndex];

					if (explicitSamplingTarget != null)
					{
						// Skip hits inside of probabilistic volumes if we're explicitely sampling a light
						// if (rec.EntityPtr->Material->Type == MaterialType.ProbabilisticVolume)
						// {
						// 	hitIndex++;
						// 	continue;
						// }

						// We explicitly sampled an entity and could not hit it, fail this sample
						if (explicitSamplingTarget != rec.EntityPtr->Content)
						{
#if PATH_DEBUGGING
							if (doDebugPaths)
								DebugPaths[depth] = new DebugPath { From = ray.Origin, To = rec.Point };
#endif
							stopTracing = true;
							break;
						}
					}

					Material* material = rec.EntityPtr->Material;

					// TODO: This is messy
					if (currentProbabilisticVolumeMaterial != null || // Inside a volume
					    material->Type == MaterialType.ProbabilisticVolume) // Entering a volume
					{
						bool isEntryHit = currentProbabilisticVolumeMaterial == null;
						if (currentProbabilisticVolumeMaterial == null)
							currentProbabilisticVolumeMaterial = material;

						// Look for an obstacle or an exit hit
						int exitHitIndex = hitIndex;
						int lastExitIndex = -1;
						int sameMaterialEntries = 0;
						while (exitHitIndex < hitRecordBuffer.Length)
						{
							HitRecord hit = hitRecordBuffer[exitHitIndex];
							if (hit.EntityPtr->Material == currentProbabilisticVolumeMaterial)
							{
								if (dot(hit.Normal, ray.Direction) < 0)
									sameMaterialEntries++;
								else
								{
									sameMaterialEntries--;
									lastExitIndex = exitHitIndex;
								}

								if (sameMaterialEntries <= 0)
									break;
							}
							else
								break;

							exitHitIndex++;
						}
						if (sameMaterialEntries > 0 && lastExitIndex != -1)
							exitHitIndex = lastExitIndex;

						if (exitHitIndex < hitRecordBuffer.Length)
						{
							HitRecord exitHitRecord = hitRecordBuffer[exitHitIndex];
							float distanceInProbabilisticVolume = exitHitRecord.Distance;
							float probabilisticVolumeEntryDistance = 0;

							if (isEntryHit)
							{
								Trace($"#{depth} : Entry hit");

								// Factor in entry distance
								probabilisticVolumeEntryDistance = rec.Distance;
								distanceInProbabilisticVolume -= rec.Distance;
							}
							else
							{
								Trace($"#{depth} : Ray origin is inside a volume");
							}

							if (currentProbabilisticVolumeMaterial->ProbabilisticHit(ref distanceInProbabilisticVolume, ref rng))
							{
								// We hit inside the volume; hijack the current hit record's distance and material
								// TODO: Normal is sort of undefined here, but should still be set
								Trace($"#{depth} : We hit inside the volume");
								float totalDistance = probabilisticVolumeEntryDistance + distanceInProbabilisticVolume;
								rec = new HitRecord(totalDistance, ray.GetPoint(totalDistance), -ray.Direction, default);
								material = currentProbabilisticVolumeMaterial;
							}
							else
							{
								// No hit inside the volume, exit it
								Trace($"#{depth} : No hit inside volume, passing through");
								currentProbabilisticVolumeMaterial = null;

								if (exitHitRecord.EntityPtr->Material->Type == MaterialType.ProbabilisticVolume &&
								    dot(exitHitRecord.Normal, ray.Direction) > 0)
								{
									// Volume exit, move to next hit
									Trace($"#{depth} : Volume exit, advance to next hit");
									hitIndex = exitHitIndex + 1;
									continue;
								}

								// Obstacle, continue
								Trace($"#{depth} : Obstacle, scattering on exit hit");
								rec = exitHitRecord;
								material = rec.EntityPtr->Material;
							}
						}
						else
						{
							// No more surfaces to hit (probabilistic volume has holes)
							Trace($"#{depth} : No more surfaces to hit, breaking");
							hitRecordBuffer.Clear();
							break;
						}
					}
#if PATH_DEBUGGING
					if (doDebugPaths)
						DebugPaths[depth] = new DebugPath { From = ray.Origin, To = rec.Point };
#endif
					material->Scatter(ray, rec, ref rng, PerlinNoise, out float3 albedo, out Ray scatteredRay);

					float3 emission = material->Emit(rec.TexCoords, PerlinNoise);
					*emissionCursor++ = emission;

					if (depth == 0)
						sampleNormal = rec.Normal;

					if (!firstNonSpecularHit)
					{
						if (material->IsPerfectSpecular)
						{
							// TODO: Fresnel mix for dielectric, first diffuse bounce for metallic
						}
						else
						{
							sampleAlbedo = emission + albedo;
							sampleNormal = rec.Normal;
							firstNonSpecularHit = true;
						}
					}

					*attenuationCursor++ = albedo;
					ray = scatteredRay;

					ray = ray.OffsetTowards(dot(scatteredRay.Direction, rec.Normal) >= 0 ? rec.Normal : -rec.Normal);
					break;
				}

				// When explicit target sampling succeeded
				if (stopTracing)
					break;

				// No hit?
				if (hitIndex >= hitRecordBuffer.Length)
				{
					Trace($"#{depth} : No more hits, hitting sky");
#if PATH_DEBUGGING
					if (doDebugPaths)
						DebugPaths[depth] = new DebugPath { From = ray.Origin, To = ray.Direction * 99999 };
#endif
					// Sample the sky color
					float3 hitSkyColor = default;
					switch (Environment.SkyType)
					{
						case SkyType.GradientSky:
							hitSkyColor = lerp(Environment.SkyBottomColor, Environment.SkyTopColor, 0.5f * (ray.Direction.y + 1));
							break;

						case SkyType.CubeMap:
							hitSkyColor = Environment.SkyCubemap.Sample(ray.Direction);
							break;
					}

					*emissionCursor++ = hitSkyColor;
					*attenuationCursor++ = 1;

					if (!firstNonSpecularHit)
					{
						sampleAlbedo = hitSkyColor;
						sampleNormal = -ray.Direction;
					}

					// Stop tracing
					break;
				}
			}

			sampleColor = 0;

			// Safety : if we don't hit an emissive surface within the trace depth limit, fail this sample
			if (depth == TraceDepth)
				return false;

			// Attenuate colors from the tail of the hit stack to the head
			while (emissionCursor != emissionStack)
			{
				var a = *--attenuationCursor;
				if (any(isnan(a)))
					Debug.Log("nan attenuation (unstack)");

				var e = *--emissionCursor;
				if (any(isnan(e)))
					Debug.Log("nan emmision (unstack)");

				sampleColor *= a;
				sampleColor += e;
			}

			return true;
		}

		static void FindHitCandidates(Ray ray, BvhNode* bvhRoot, ref HybridPtrStack<BvhNode> nodeTraversalBuffer, ref HybridPtrStack<Entity> hitCandidateBuffer
#if FULL_DIAGNOSTICS
			, ref Diagnostics diagnostics
#endif
			)
		{
			float3 rayInvDirection = rcp(ray.Direction);

			// Convert NaN to INFINITY, since Burst thinks that divisions by 0 = NaN
			rayInvDirection = select(rayInvDirection, INFINITY, isnan(rayInvDirection));

			nodeTraversalBuffer.Clear();
			hitCandidateBuffer.Clear();

			nodeTraversalBuffer.Push(bvhRoot);

			// Traverse the BVH and record candidate leaf nodes
			while (nodeTraversalBuffer.Length > 0)
			{
				BvhNode* nodePtr = nodeTraversalBuffer.Pop();

				if (!nodePtr->Bounds.Hit(ray.Origin, rayInvDirection))
					continue;

#if FULL_DIAGNOSTICS
				diagnostics.BoundsHitCount++;
#endif

				if (nodePtr->IsLeaf)
				{
					int entityCount = nodePtr->EntityCount;
					Entity* entityPtr = nodePtr->EntitiesStart;
					for (int i = 0; i < entityCount; i++, ++entityPtr)
						hitCandidateBuffer.Push(entityPtr);

#if FULL_DIAGNOSTICS
					diagnostics.CandidateCount += entityCount;
#endif
				}
				else
				{
					nodeTraversalBuffer.Push(nodePtr->Left);
					nodeTraversalBuffer.Push(nodePtr->Right);
				}
			}
		}

		static void FindHits(Ray ray, ref HybridPtrStack<Entity> hitCandidateBuffer, ref HybridList<HitRecord> hitBuffer)
		{
			hitBuffer.Clear();

			while (hitCandidateBuffer.Length > 0)
			{
				Entity* hitCandidate = hitCandidateBuffer.Pop();
				if (hitCandidate->Hit(ray, 0, float.PositiveInfinity, out HitRecord thisRec))
				{
					thisRec.EntityPtr = hitCandidate;
					hitBuffer.Add(thisRec);

					// Inject exit hits for probabilistic convex hulls
					if (hitCandidate->Material->Type == MaterialType.ProbabilisticVolume &&
					    hitCandidate->Type.IsConvexHull() &&
					    hitCandidate->Hit(ray, thisRec.Distance + 0.001f, float.PositiveInfinity, out HitRecord exitRec))
					{
						exitRec.EntityPtr = hitCandidate;
						hitBuffer.Add(exitRec);
					}
				}
			}

			if (hitBuffer.Length > 0)
				hitBuffer.Sort(new HitRecord.DistanceComparer());
		}

		static void DetermineVolumeContainment(Ray ray, BvhNode* bvhRoot, ref HybridPtrStack<BvhNode> nodeTraversalBuffer, ref HybridPtrStack<Entity> hitCandidateBuffer, HybridList<HitRecord> hitBuffer,
#if FULL_DIAGNOSTICS
			ref Diagnostics diagnostics,
#endif
			out Material* volumeMaterial)
		{
			for (int i = 0; i < hitBuffer.Length; i++)
			{
				HitRecord hit = hitBuffer[i];
				if (hit.EntityPtr->Material->Type == MaterialType.ProbabilisticVolume)
				{
					// Entry hit, early out
					if (dot(hit.Normal, ray.Direction) < 0)
						break;

					// Exit hit before an entry hit, we are likely inside this volume; throw a ray backwards to make sure
					var backwardsRay = new Ray(ray.Origin, -ray.Direction, ray.Time);
					FindHitCandidates(backwardsRay, bvhRoot, ref nodeTraversalBuffer, ref hitCandidateBuffer
#if FULL_DIAGNOSTICS
						, ref diagnostics
#endif
					);
					if (AnyBackwardsVolumeEntryHit(backwardsRay, ref hitCandidateBuffer))
					{
						volumeMaterial = hit.EntityPtr->Material;
						return;
					}
				}
			}

			volumeMaterial = null;
		}

		static bool AnyBackwardsVolumeEntryHit(Ray backwardsRay, ref HybridPtrStack<Entity> hitCandidateBuffer)
		{
			for (int i = 0; i < hitCandidateBuffer.Length; i++)
			{
				Entity* hitCandidate = hitCandidateBuffer[i];
				if (hitCandidate->Material->Type == MaterialType.ProbabilisticVolume &&
				    hitCandidate->Hit(backwardsRay, 0, float.PositiveInfinity, out HitRecord hitRecord) &&
				    dot(hitRecord.Normal, backwardsRay.Direction) > 0)
				{
					return true;
				}
			}

			return false;
		}

		[BurstDiscard]
		[Conditional("TRACE_LOGGING")]
		void Trace(string text)
		{
			Debug.Log(text);
		}
	}
}