using System;
using System.Diagnostics;
using OpenImageDenoise;
using Runtime.EntityTypes;
using Unity;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Util;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;
using Debug = UnityEngine.Debug;

#if ENABLE_OPTIX
using OptiX;
#endif

namespace Runtime
{
	struct Diagnostics
	{
#if FULL_DIAGNOSTICS
		public float RayCount;
		public float BoundsHitCount;
		public float CandidateCount;
#pragma warning disable 649
		public float Padding;
#pragma warning restore 649
#else
		public float RayCount;
#endif
	}

#if PATH_DEBUGGING
	struct DebugPath
	{
		public float3 From, To;
	}
#endif

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	unsafe struct AddMeshRuntimeEntitiesJob : IJob
	{
		[ReadOnly] public Mesh.MeshDataArray MeshDataArray;
		[ReadOnly] [NativeDisableUnsafePtrRestriction] public Material* Material;

		[WriteOnly] public NativeArray<Triangle> Triangles;
		[WriteOnly] public NativeArray<Entity> Entities;
		[WriteOnly] public UnsafePtrList<Entity> ImportanceSampledEntityPointers;

		public NativeReference<int> TriangleIndex, EntityIndex;

		public bool FaceNormals, Moving;
		public RigidTransform RigidTransform;
		public float3 DestinationOffset;
		public float2 TimeRange;
		public float Scale;

		public void Execute()
		{
			for (int meshIndex = 0; meshIndex < MeshDataArray.Length; meshIndex++)
			{
				Mesh.MeshData meshData = MeshDataArray[meshIndex];

				using var vertices = new NativeArray<Vector3>(meshData.vertexCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
				meshData.GetVertices(vertices);

				NativeArray<Vector3> normals = default;
				if (!FaceNormals)
				{
					normals = new NativeArray<Vector3>(meshData.vertexCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
					meshData.GetNormals(normals);
				}

				NativeArray<ushort> indices = meshData.GetIndexData<ushort>();
				for (int i = 0; i < indices.Length; i += 3)
				{
					int3 triangleIndices = int3(indices[i], indices[i + 1], indices[i + 2]);

					// Bake transform
					float3* worldSpaceVertices = stackalloc float3[3];
					float3* worldSpaceNormals = stackalloc float3[3];
					for (int j = 0; j < 3; j++)
					{
						worldSpaceVertices[j] = transform(RigidTransform, vertices[triangleIndices[j]]) * Scale;
						if (!FaceNormals) worldSpaceNormals[j] = mul(RigidTransform.rot, normals[triangleIndices[j]]);
					}

					if (FaceNormals)
						Triangles[TriangleIndex.Value] = new Triangle(worldSpaceVertices[0], worldSpaceVertices[1], worldSpaceVertices[2]);
					else
						Triangles[TriangleIndex.Value] = new Triangle(
							worldSpaceVertices[0], worldSpaceVertices[1], worldSpaceVertices[2],
							worldSpaceNormals[0], worldSpaceNormals[1], worldSpaceNormals[2]);

					var contentPointer = (Triangle*) Triangles.GetUnsafePtr() + TriangleIndex.Value++;

					Entity entity = Moving
						? new Entity(EntityType.Triangle, contentPointer, default, Material, true, DestinationOffset, TimeRange)
						: new Entity(EntityType.Triangle, contentPointer, default, Material);

					Entities[EntityIndex.Value++] = entity;

					if (Material->Type == MaterialType.DiffuseLight)
						ImportanceSampledEntityPointers.AddNoResize((Entity*) Entities.GetUnsafePtr() + (EntityIndex.Value - 1));
				}

				if (!FaceNormals)
					normals.Dispose();
			}
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct BuildRuntimeBvhJob : IJob
	{
		[ReadOnly] public NativeList<BvhNodeData> BvhNodeDataBuffer;
		[WriteOnly] public NativeArray<BvhNode> BvhNodeBuffer;
		public int NodeCount;

		int nodeIndex;

		// Runtime BVH is inserted BACKWARDS while traversing postorder, which means the first node will be the root

		unsafe BvhNode* WalkBvh(BvhNodeData* nodeData)
		{
			BvhNode* leftNode = null, rightNode = null;

			if (!nodeData->IsLeaf)
			{
				leftNode = WalkBvh(nodeData->Left);
				rightNode = WalkBvh(nodeData->Right);
			}

			BvhNodeBuffer[nodeIndex] = new BvhNode(nodeData->Bounds, nodeData->EntitiesStart, nodeData->EntityCount,
				leftNode, rightNode);
			return (BvhNode*) BvhNodeBuffer.GetUnsafePtr() + nodeIndex--;
		}

		public unsafe void Execute()
		{
			nodeIndex = NodeCount - 1;
			WalkBvh((BvhNodeData*) BvhNodeDataBuffer.GetUnsafeReadOnlyPtr());
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	unsafe struct AccumulateJob : IJobParallelFor
	{
		const int BvhNodeStackAllocSize = 128;
		const int EntityStackAllocSize = 64;
		const int HitRecordStackAllocSize = 32;

		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public float2 Size;
		[ReadOnly] public int SliceOffset;
		[ReadOnly] public int SliceDivider;
		[ReadOnly] public uint Seed;
		[ReadOnly] public View View;
		[ReadOnly] public Environment Environment;
		[ReadOnly] public uint SampleCount;
		[ReadOnly] public int TraceDepth;
		[ReadOnly] public bool SubPixelJitter;
		[ReadOnly] public ImportanceSampler ImportanceSampler;
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
			if (CancellationToken[0])
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
				float2 normalizedCoordinates = (coordinates + (SubPixelJitter ? blueNoise.NextFloat2() : 0.5f)) / Size;
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
								rec = new HitRecord(totalDistance, ray.GetPoint(totalDistance), -ray.Direction);
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

					float3 emission = material->Emit(rec.Point, rec.Normal, PerlinNoise);
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
							sampleAlbedo = material->Type == MaterialType.DiffuseLight ? emission : albedo;
							sampleNormal = rec.Normal;
							firstNonSpecularHit = true;
						}
					}

					if (ImportanceSampler.Mode == ImportanceSamplingMode.None || material->IsPerfectSpecular)
					{
						*attenuationCursor++ = albedo;
						ray = scatteredRay;
					}
					else
					{
						float3 outgoingLightDirection = -ray.Direction;
						float scatterPdfValue = material->Pdf(scatteredRay.Direction, outgoingLightDirection, rec.Normal);

						// We reached our target, stop tracing
						if (explicitSamplingTarget != null)
							stopTracing = true;

						ImportanceSampler.Sample(scatteredRay, outgoingLightDirection, rec, material, ref rng,
							out ray, out float pdfValue, out explicitSamplingTarget);

						// We already hit the target, stop tracing
						stopTracing |= explicitSamplingTarget != null && rec.EntityPtr != null && explicitSamplingTarget == rec.EntityPtr->Content;

						// Ignore probabilistic volume containment when doing explicit light sampling
						currentProbabilisticVolumeMaterial = null;

						// Scatter ray is likely parallel to the surface, and division would cause a NaN
						if (pdfValue.AlmostEquals(0))
						{
							*attenuationCursor++ = 1;
							break;
						}

						*attenuationCursor++ = albedo * scatterPdfValue / pdfValue;
					}

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

				if (!nodePtr->Bounds.Hit(ray.Origin, rayInvDirection, 0, float.PositiveInfinity))
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
		[Conditional("PATH_DEBUGGING")]
		private void Trace(string text)
		{
			Debug.Log(text);
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct CombineJob : IJobParallelFor
	{
		static readonly float3 NoSamplesColor = new float3(1, 0, 1);
		static readonly float3 NaNColor = new float3(0, 1, 1);

		public bool DebugMode;
		public bool LdrAlbedo;

		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float4> InputColor;
		[ReadOnly] public NativeArray<float3> InputNormal;
		[ReadOnly] public NativeArray<float3> InputAlbedo;
		[ReadOnly] public int2 Size;

		[WriteOnly] public NativeArray<float3> OutputColor;
		[WriteOnly] public NativeArray<float3> OutputNormal;
		[WriteOnly] public NativeArray<float3> OutputAlbedo;

		public void Execute(int index)
		{
			if (CancellationToken[0])
				return;

			float4 inputColor = InputColor[index];
			var realSampleCount = (int) inputColor.w;

			float3 finalColor;
			if (!DebugMode)
			{
				if (realSampleCount == 0)
				{
					int tentativeIndex = index;

					// look-around (for interlaced buffer)
					while (realSampleCount == 0 && (tentativeIndex -= Size.x) >= 0)
					{
						inputColor = InputColor[tentativeIndex];
						realSampleCount = (int) inputColor.w;
					}
				}

				if (realSampleCount == 0) finalColor = 0;
				else if (any(isnan(inputColor))) finalColor = 0;
				else finalColor = inputColor.xyz / realSampleCount;
			}
			else
			{
				if (realSampleCount == 0) finalColor = NoSamplesColor;
				else if (any(isnan(inputColor))) finalColor = NaNColor;
				else finalColor = inputColor.xyz / realSampleCount;
			}

			float3 finalAlbedo = InputAlbedo[index] / max(realSampleCount, 1);

			if (LdrAlbedo)
				finalAlbedo = min(finalAlbedo, 1);

			OutputColor[index] = finalColor;
			OutputNormal[index] = normalizesafe(InputNormal[index] / max(realSampleCount, 1));
			OutputAlbedo[index] = finalAlbedo;
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct ClearBufferJob<T> : IJob where T : unmanaged
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[WriteOnly] public NativeArray<T> Buffer;

		public void Execute()
		{
			if (CancellationToken[0])
				return;

			Buffer.ZeroMemory();
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct CopyFloat3BufferJob : IJob
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> Input;
		[WriteOnly] public NativeArray<float3> Output;

		public void Execute()
		{
			if (CancellationToken[0])
				return;

			NativeArray<float3>.Copy(Input, Output);
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct CopyFloat4BufferJob : IJob
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float4> Input;
		[WriteOnly] public NativeArray<float4> Output;

		public void Execute()
		{
			if (CancellationToken[0])
				return;

			NativeArray<float4>.Copy(Input, Output);
		}
	}

	// because the OIDN API uses strings, we can't use Burst here
	struct OpenImageDenoiseJob : IJob
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> InputColor, InputNormal, InputAlbedo;
		[ReadOnly] public ulong Width, Height;

		public NativeArray<float3> OutputColor;

		public OidnFilter DenoiseFilter;

		public unsafe void Execute()
		{
			if (CancellationToken[0])
				return;

			OidnFilter.SetSharedImage(DenoiseFilter, "color", new IntPtr(InputColor.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);
			OidnFilter.SetSharedImage(DenoiseFilter, "normal", new IntPtr(InputNormal.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);
			OidnFilter.SetSharedImage(DenoiseFilter, "albedo", new IntPtr(InputAlbedo.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);

			OidnFilter.SetSharedImage(DenoiseFilter, "output", new IntPtr(OutputColor.GetUnsafePtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);

			OidnFilter.Commit(DenoiseFilter);
			OidnFilter.Execute(DenoiseFilter);
		}
	}

#if ENABLE_OPTIX
	// Disabled because it current won't compile using Burst (I swear it used to work)
	// [BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct OptixDenoiseJob : IJob
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> InputColor, InputAlbedo;
		[ReadOnly] public uint2 BufferSize;
		[ReadOnly] public OptixDenoiserSizes DenoiserSizes;

		[WriteOnly] public NativeArray<float3> OutputColor;

		public OptixDenoiser Denoiser;
		public CudaStream CudaStream;

		public CudaBuffer InputColorBuffer,
			InputAlbedoBuffer,
			OutputColorBuffer,
			ScratchMemory,
			DenoiserState;

		[BurstDiscard]
		static void Check(CudaError cudaError)
		{
			if (cudaError != CudaError.Success)
				Debug.LogError($"CUDA Error : {cudaError}");
		}

		public unsafe void Execute()
		{
			if (CancellationToken[0])
				return;

			Check(CudaBuffer.Copy(new IntPtr(InputColor.GetUnsafeReadOnlyPtr()), InputColorBuffer.Handle,
				InputColor.Length * sizeof(float3), CudaMemcpyKind.HostToDevice));
			Check(CudaBuffer.Copy(new IntPtr(InputAlbedo.GetUnsafeReadOnlyPtr()), InputAlbedoBuffer.Handle,
				InputAlbedo.Length * sizeof(float3), CudaMemcpyKind.HostToDevice));

			var colorImage = new OptixImage2D
			{
				Data = InputColorBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};
			var albedoImage = new OptixImage2D
			{
				Data = InputAlbedoBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};

			OptixImage2D* optixImages = stackalloc OptixImage2D[2];
			optixImages[0] = colorImage;
			optixImages[1] = albedoImage;

			OptixDenoiserParams denoiserParams = default;

			var outputImage = new OptixImage2D
			{
				Data = OutputColorBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};

			OptixDenoiser.Invoke(Denoiser, CudaStream, &denoiserParams, DenoiserState, DenoiserSizes.StateSizeInBytes,
				optixImages, 2, 0, 0,
				&outputImage, ScratchMemory, DenoiserSizes.RecommendedScratchSizeInBytes);

			Check(CudaBuffer.Copy(OutputColorBuffer.Handle, new IntPtr(OutputColor.GetUnsafePtr()),
				OutputColor.Length * sizeof(float3), CudaMemcpyKind.DeviceToHost));
		}
	}
#endif

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct FinalizeTexturesJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> InputColor;
		[ReadOnly] public NativeArray<float3> InputNormal;
		[ReadOnly] public NativeArray<float3> InputAlbedo;

		[WriteOnly] public NativeArray<RGBA32> OutputColor;
		[WriteOnly] public NativeArray<RGBA32> OutputNormal;
		[WriteOnly] public NativeArray<RGBA32> OutputAlbedo;

		public void Execute(int index)
		{
			if (CancellationToken[0])
				return;

			// TODO: tone-mapping
			float3 outputColor = saturate(InputColor[index].LinearToGamma()) * 255;
			OutputColor[index] = new RGBA32
			{
				r = (byte) outputColor.x,
				g = (byte) outputColor.y,
				b = (byte) outputColor.z,
				a = 255
			};

			float3 outputNormal = saturate((InputNormal[index] * 0.5f + 0.5f).LinearToGamma()) * 255;
			OutputNormal[index] = new RGBA32
			{
				r = (byte) outputNormal.x,
				g = (byte) outputNormal.y,
				b = (byte) outputNormal.z,
				a = 255
			};

			float3 outputAlbedo = saturate(InputAlbedo[index].LinearToGamma()) * 255;
			OutputAlbedo[index] = new RGBA32
			{
				r = (byte) outputAlbedo.x,
				g = (byte) outputAlbedo.y,
				b = (byte) outputAlbedo.z,
				a = 255
			};
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct ReduceRayCountJob : IJob
	{
		[ReadOnly] public NativeArray<Diagnostics> Diagnostics;
		[WriteOnly] public NativeReference<int> TotalRayCount;

		public void Execute()
		{
			float totalRayCount = 0;

			for (int i = 0; i < Diagnostics.Length; i++)
				totalRayCount += Diagnostics[i].RayCount;

			TotalRayCount.Value = (int) totalRayCount;
		}
	}

	struct RecordTimeJob : IJob
	{
		[ReadOnly] public int Index;
		[WriteOnly] public NativeArray<long> Buffer;

		public void Execute()
		{
			Buffer[Index] = Stopwatch.GetTimestamp();
		}
	}
}