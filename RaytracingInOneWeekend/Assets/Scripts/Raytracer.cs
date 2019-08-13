using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions.Must;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;
using Debug = UnityEngine.Debug;
using float3 = Unity.Mathematics.float3;
using quaternion = Unity.Mathematics.quaternion;
using Random = Unity.Mathematics.Random;
using RigidTransform = Unity.Mathematics.RigidTransform;
#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
using OdinReadOnly = Sirenix.OdinInspector.ReadOnlyAttribute;
#else
using OdinMock;
using OdinReadOnly = OdinMock.ReadOnlyAttribute;
#endif

namespace RaytracerInOneWeekend
{
	unsafe partial class Raytracer : MonoBehaviour
	{
		[Title("References")]
		[SerializeField] UnityEngine.Camera targetCamera = null;

		[Title("Settings")]
		[SerializeField] [Range(0.01f, 2)] float resolutionScaling = 0.5f;
		[SerializeField] [Range(1, 10000)] uint samplesPerPixel = 1000;
		[SerializeField] [Range(1, 100)] uint samplesPerBatch = 10;
		[SerializeField] [Range(1, 100)] int traceDepth = 35;
		[SerializeField] bool previewAfterBatch = true;
		[SerializeField] bool stopWhenCompleted = true;

		[Title("World")]
		[InlineEditor(DrawHeader = false)]
		[SerializeField] internal SceneData scene = null;

		[Title("Debug")]
		[SerializeField] Shader viewRangeShader = null;
		[OdinReadOnly] public uint AccumulatedSamples;

		[UsedImplicitly]
		[OdinReadOnly]
		public float MillionRaysPerSecond, AvgMRaysPerSecond, LastBatchDuration, LastTraceDuration;

		[UsedImplicitly]
		[OdinReadOnly]
		public float BufferMinValue, BufferMaxValue;

		UnityEngine.Material viewRangeMaterial;
		Texture2D frontBufferTexture, diagnosticsTexture;
		CommandBuffer commandBuffer;

		// TODO: allocation order of all these buffers is probably pretty crucial for cache locality

		NativeArray<RGBA32> frontBuffer;
		NativeArray<Diagnostics> diagnosticsBuffer;
		NativeArray<float4> accumulationInputBuffer, accumulationOutputBuffer;

#if SOA_SIMD
		SoaSpheres sphereBuffer;
		SoaSpheres World => sphereBuffer;
#elif AOSOA_SIMD
		AosoaSpheres sphereBuffer;
		AosoaSpheres World => sphereBuffer;
#else // !SOA_SIMD && !AOSOA_SIMD
		NativeArray<Sphere> sphereBuffer;
		NativeArray<Rect> rectBuffer;
		NativeArray<Entity> entityBuffer;
#if !BVH
		NativeArray<Entity> World => entityBuffer;
#endif
#endif

#if BVH
		NativeList<BvhNode> bvhNodeBuffer;
		NativeList<BvhNodeMetadata> bvhNodeMetadataBuffer;
		BvhNode* World => bvhNodeBuffer.IsCreated ? (BvhNode*) bvhNodeBuffer.GetUnsafePtr() : null;
#endif

		readonly PerlinDataGenerator perlinData = new PerlinDataGenerator();

		JobHandle? accumulateJobHandle;
		JobHandle? combineJobHandle;

		bool commandBufferHooked;
		bool worldNeedsRebuild;
		bool initialized;
		float focusDistance;
		bool traceAborted;
		bool ignoreBatchTimings;
		int lastTraceDepth;

		readonly Stopwatch batchTimer = new Stopwatch();
		readonly Stopwatch traceTimer = new Stopwatch();
		readonly List<float> mraysPerSecResults = new List<float>();

		internal readonly List<EntityData> activeEntities = new List<EntityData>();
		readonly List<MaterialData> activeMaterials = new List<MaterialData>();

		float2 bufferSize;

		bool TraceActive => accumulateJobHandle.HasValue || combineJobHandle.HasValue;

		enum BufferView
		{
			Front,
			RayCount,
#if FULL_DIAGNOSTICS
			BvhHitCount,
			CandidateCount
#endif
		}

#if !UNITY_EDITOR
		const BufferView bufferView = BufferView.Front;
#endif
		int channelPropertyId, minimumRangePropertyId;

		void Awake()
		{
			commandBuffer = new CommandBuffer { name = "Raytracer" };

			const HideFlags flags = HideFlags.HideAndDontSave;
			frontBufferTexture = new Texture2D(0, 0, TextureFormat.RGBA32, false) { hideFlags = flags };
#if FULL_DIAGNOSTICS && BVH_ITERATIVE
			diagnosticsTexture = new Texture2D(0, 0, TextureFormat.RGBAFloat, false) { hideFlags = flags };
#else
			diagnosticsTexture = new Texture2D(0, 0, TextureFormat.RFloat, false) { hideFlags = flags };
#endif

			viewRangeMaterial = new UnityEngine.Material(viewRangeShader);
			channelPropertyId = Shader.PropertyToID("_Channel");
			minimumRangePropertyId = Shader.PropertyToID("_Minimum_Range");

			ignoreBatchTimings = true;
		}

		void Start()
		{
			targetCamera.RemoveAllCommandBuffers();

	#if UNITY_EDITOR
			scene = scene.DeepClone();
	#endif

			RebuildWorld();
			EnsureBuffersBuilt();
			CleanCamera();

			ScheduleAccumulate(true);
		}

		void OnDestroy()
		{
			// if there is a running job, let it know it needs to cancel and wait for completion
			accumulateJobHandle?.Complete();
			combineJobHandle?.Complete();

#if SOA_SIMD || AOSOA_SIMD
			sphereBuffer.Dispose(); // this actually isn't a NativeArray<T>!
#else
			entityBuffer.SafeDispose();
			sphereBuffer.SafeDispose();
			rectBuffer.SafeDispose();
#endif
			accumulationInputBuffer.SafeDispose();
			accumulationOutputBuffer.SafeDispose();
#if BVH
			bvhNodeBuffer.SafeDispose();
			bvhNodeMetadataBuffer.SafeDispose();
#endif
			perlinData.Dispose();

#if UNITY_EDITOR
			if (scene.hideFlags == HideFlags.HideAndDontSave)
				Destroy(scene);
#endif
		}

		void Update()
		{
#if UNITY_EDITOR
			WatchForWorldChanges();
#endif
			uint2 currentSize = uint2(
				(uint) ceil(targetCamera.pixelWidth * resolutionScaling),
				(uint) ceil(targetCamera.pixelHeight * resolutionScaling));

			bool buffersNeedRebuild = any(currentSize != bufferSize);
			bool cameraDirty = targetCamera.transform.hasChanged;
			bool traceDepthChanged = traceDepth != lastTraceDepth;

			bool traceNeedsReset = buffersNeedRebuild || worldNeedsRebuild || cameraDirty || traceDepthChanged;
			bool traceNeedsKick = traceNeedsReset || !commandBufferHooked;

			void RebuildDirtyComponents()
			{
				if (buffersNeedRebuild) EnsureBuffersBuilt();
				if (worldNeedsRebuild) RebuildWorld();
				if (cameraDirty) CleanCamera();
			}

			void CompleteAccumulate()
			{
				TimeSpan elapsedTime = batchTimer.Elapsed;

				accumulateJobHandle.Value.Complete();
				accumulateJobHandle = null;

				float totalRayCount = diagnosticsBuffer.Sum(x => x.RayCount);

				AccumulatedSamples += samplesPerBatch;
				LastBatchDuration = (float) elapsedTime.TotalMilliseconds;
				MillionRaysPerSecond = totalRayCount / (float) elapsedTime.TotalSeconds / 1000000;
				if (!ignoreBatchTimings) mraysPerSecResults.Add(MillionRaysPerSecond);
				AvgMRaysPerSecond = mraysPerSecResults.Count == 0 ? 0 : mraysPerSecResults.Average();
				ignoreBatchTimings = false;
			}

			if (!TraceActive && traceNeedsKick)
			{
				RebuildDirtyComponents();
				ScheduleAccumulate(traceNeedsReset);
			}

			if (combineJobHandle.HasValue && combineJobHandle.Value.IsCompleted)
			{
				if (accumulateJobHandle.HasValue && accumulateJobHandle.Value.IsCompleted)
					CompleteAccumulate();

				combineJobHandle.Value.Complete();
				combineJobHandle = null;

				bool traceCompleted = false;
				if (AccumulatedSamples >= samplesPerPixel)
				{
					traceCompleted = true;
					LastTraceDuration = (float) traceTimer.Elapsed.TotalMilliseconds;
				}

				SwapBuffers();
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
				RebuildDirtyComponents();

				if ((!(traceCompleted && stopWhenCompleted) || traceNeedsReset) && !traceAborted)
					ScheduleAccumulate(traceCompleted | traceNeedsReset);

				traceAborted = false;
			}

			// only when preview is disabled
			if (!combineJobHandle.HasValue && accumulateJobHandle.HasValue && accumulateJobHandle.Value.IsCompleted)
			{
				CompleteAccumulate();
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
				RebuildDirtyComponents();

				if (!traceAborted)
					ScheduleAccumulate(false);

				traceAborted = false;
			}
		}

		void CleanCamera()
		{
#if UNITY_EDITOR
			targetCamera.transform.hasChanged = false;
			scene.UpdateFromGameView();
#endif
		}

		void SwapBuffers()
		{
			float bufferMin = float.MaxValue, bufferMax = float.MinValue;

			switch (bufferView)
			{
				case BufferView.RayCount:
					foreach (Diagnostics value in diagnosticsBuffer)
					{
						bufferMin = min(bufferMin, value.RayCount);
						bufferMax = max(bufferMax, value.RayCount);
					}
					break;

#if FULL_DIAGNOSTICS && BVH_ITERATIVE
				case BufferView.BvhHitCount:
					foreach (Diagnostics value in diagnosticsBuffer)
					{
						bufferMin = min(bufferMin, value.BoundsHitCount);
						bufferMax = max(bufferMax, value.BoundsHitCount);
					}
					break;

				case BufferView.CandidateCount:
					foreach (Diagnostics value in diagnosticsBuffer)
					{
						bufferMin = min(bufferMin, value.CandidateCount);
						bufferMax = max(bufferMax, value.CandidateCount);
					}
					break;
#endif
			}

			switch (bufferView)
			{
				case BufferView.Front: frontBufferTexture.Apply(false); break;
				default:
					BufferMinValue = bufferMin;
					BufferMaxValue = bufferMax;
					diagnosticsTexture.Apply(false);
					viewRangeMaterial.SetVector(minimumRangePropertyId, new Vector4(bufferMin, bufferMax - bufferMin));
					break;
			}

			if (!commandBufferHooked)
			{
				commandBuffer.Clear();

				var blitTarget = new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget);
				switch (bufferView)
				{
					case BufferView.Front:
						commandBuffer.Blit(frontBufferTexture, blitTarget);
						break;

					default:
						viewRangeMaterial.SetInt(channelPropertyId, (int) bufferView - 1);
						commandBuffer.Blit(diagnosticsTexture, blitTarget, viewRangeMaterial);
						break;
				}

				targetCamera.AddCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
				commandBufferHooked = true;
			}
		}

		void ScheduleAccumulate(bool firstBatch)
		{
			Transform cameraTransform = targetCamera.transform;
			Vector3 origin = cameraTransform.localPosition;
			Vector3 lookAt = origin + cameraTransform.forward;

			if (HitWorld(new Ray(origin, cameraTransform.forward), out HitRecord hitRec))
				focusDistance = hitRec.Distance;

			var raytracingCamera = new Camera(origin, lookAt, cameraTransform.up, scene.CameraFieldOfView,
				bufferSize.x / bufferSize.y, scene.CameraAperture, focusDistance, 0, 1);

			var totalBufferSize = (int) (bufferSize.x * bufferSize.y);

			if (firstBatch)
			{
				accumulationInputBuffer.SafeDispose();
				accumulationInputBuffer = new NativeArray<float4>(totalBufferSize, Allocator.Persistent);

				mraysPerSecResults.Clear();
				AccumulatedSamples = 0;
				lastTraceDepth = traceDepth;
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
			}
			else
				Util.Swap(ref accumulationInputBuffer, ref accumulationOutputBuffer);

			var accumulateJob = new AccumulateJob
			{
				Size = bufferSize,
				Camera = raytracingCamera,
				SkyBottomColor = scene.SkyBottomColor.ToFloat3(),
				SkyTopColor = scene.SkyTopColor.ToFloat3(),
				InputSamples = accumulationInputBuffer,
				Seed = (uint) Time.frameCount + 1,
				SampleCount = min(samplesPerPixel, samplesPerBatch),
				TraceDepth = traceDepth,
				World = World,
				PerlinData = perlinData.GetRuntimeData(),
				OutputSamples = accumulationOutputBuffer,
				OutputDiagnostics = diagnosticsBuffer,
#if BVH_ITERATIVE
				NodeCount = bvhNodeBuffer.Length,
				EntityCount = entityBuffer.Length,
#endif
			};

			accumulateJobHandle = accumulateJob.Schedule(totalBufferSize, 1);

			if (AccumulatedSamples + samplesPerBatch >= samplesPerPixel || previewAfterBatch)
			{
				var combineJob = new CombineJob { Input = accumulationOutputBuffer, Output = frontBuffer };
				combineJobHandle = combineJob.Schedule(totalBufferSize, 128, accumulateJobHandle.Value);
			}

			batchTimer.Restart();
			if (firstBatch) traceTimer.Restart();
			JobHandle.ScheduleBatchedJobs();
		}

		void EnsureBuffersBuilt()
		{
			int width = (int) ceil(targetCamera.pixelWidth * resolutionScaling);
			int height = (int) ceil(targetCamera.pixelHeight * resolutionScaling);

			if (frontBufferTexture.width != width || frontBufferTexture.height != height)
			{
				if (commandBufferHooked)
				{
					targetCamera.RemoveCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
					commandBufferHooked = false;
				}

				void PrepareTexture<T>(Texture2D texture, out NativeArray<T> buffer) where T : struct
				{
					texture.Resize(width, height);
					texture.filterMode = resolutionScaling > 1 ? FilterMode.Bilinear : FilterMode.Point;
					buffer = texture.GetRawTextureData<T>();
				}

				PrepareTexture(frontBufferTexture, out frontBuffer);
				PrepareTexture(diagnosticsTexture, out diagnosticsBuffer);

				Debug.Log($"Rebuilt front buffer (now {width} x {height})");
			}

			if (accumulationOutputBuffer.EnsureCapacity(width * height))
				Debug.Log($"Rebuilt accumulation output buffer (now {width} x {height})");

			bufferSize = float2(width, height);
		}

		void RebuildWorld()
		{
#if UNITY_EDITOR
			if (scene) scene.ClearDirty();
#endif
			CollectActiveEntities();

			activeMaterials.Clear();
			foreach (EntityData entity in activeEntities)
				if (!activeMaterials.Contains(entity.Material))
					activeMaterials.Add(entity.Material);

#if SOA_SIMD || AOSOA_SIMD
			int sphereCount = activeEntities.Count(x => x.Type == EntityType.Sphere);
			if (sphereBuffer.Length != sphereCount)
			{
				sphereBuffer.Dispose();
#if SOA_SIMD
				sphereBuffer = new SoaSpheres(sphereCount);
#elif AOSOA_SIMD
				sphereBuffer = new AosoaSpheres(sphereCount);
#endif
			}

			int i = 0;
			foreach (var e in activeEntities.Where(x => x.Type == EntityType.Sphere))
			{
				var s = e.SphereData;
				sphereBuffer.SetElement(i, s.CenterFrom, s.CenterTo, s.FromTime, s.ToTime, s.Radius);

				MaterialData material = e.Material;
				TextureData albedo = material ? material.Albedo : null;
				TextureData emission = material ? material.Emission : null;

				sphereBuffer.Material[i] =
					new Material(material.Type, material.TextureScale * s.Radius, albedo.GetRuntimeData(),
						emission.GetRuntimeData(), material.Fuzz, material.RefractiveIndex);

				i++;
			}

#else // !SOA_SIMD && !AOSOA_SIMD
			RebuildEntityBuffers();
#endif

#if BVH
			RebuildBvh();
#endif

			perlinData.Generate(scene.RandomSeed);

			worldNeedsRebuild = false;
		}

#if !SOA_SIMD && !AOSOA_SIMD
		void RebuildEntityBuffers()
		{
			int entityCount = activeEntities.Count;

			entityBuffer.EnsureCapacity(entityCount);

			sphereBuffer.EnsureCapacity(activeEntities.Count(x => x.Type == EntityType.Sphere));
			rectBuffer.EnsureCapacity(activeEntities.Count(x => x.Type == EntityType.Rect));

			int entityIndex = 0, sphereIndex = 0, rectIndex = 0;
			foreach (EntityData e in activeEntities)
			{
				Vector2 sizeFactor = Vector2.one;
				void* contentPointer = null;
				RigidTransform rigidTransform = new RigidTransform(e.Rotation, e.Position);

				switch (e.Type)
				{
					case EntityType.Sphere:
						SphereData s = e.SphereData;
						sphereBuffer[sphereIndex] = new Sphere(s.Radius);
						sizeFactor *= s.Radius;
						contentPointer = (Sphere*) sphereBuffer.GetUnsafePtr() + sphereIndex;
						sphereIndex++;
						break;

					case EntityType.Rect:
						RectData r = e.RectData;
						rectBuffer[rectIndex] = new Rect(r.Size);
						sizeFactor *= r.Size;
						contentPointer = (Rect*) rectBuffer.GetUnsafePtr() + rectIndex;
						rectIndex++;
						break;
				}

				MaterialData materialData = e.Material;
				TextureData albedo = materialData ? materialData.Albedo : null;
				TextureData emission = materialData ? materialData.Emission : null;

				Material material = materialData
					? new Material(materialData.Type, materialData.TextureScale * sizeFactor,
						albedo.GetRuntimeData(), emission.GetRuntimeData(),
						materialData.Fuzz, materialData.RefractiveIndex)
					: default;

				entityBuffer[entityIndex++] = new Entity(e.Type, contentPointer, rigidTransform, material);
			}
		}
#endif // !SOA_SIMD && !AOSOA_SIMD

#if BVH
		void RebuildBvh()
		{
			int nodeCount = BvhNode.GetNodeCount(entityBuffer);
			bvhNodeBuffer.EnsureCapacity(nodeCount);
			bvhNodeMetadataBuffer.EnsureCapacity(nodeCount);

			bvhNodeBuffer.Clear();
			bvhNodeMetadataBuffer.Clear();

			var rootNode = new BvhNode(entityBuffer, bvhNodeBuffer, bvhNodeMetadataBuffer);
			rootNode.Metadata->Id = bvhNodeBuffer.Length;
			bvhNodeBuffer.Add(rootNode);

			bvhNodeBuffer.AsArray().Sort(new BvhNodeComparer());
			World->SetupPointers(bvhNodeBuffer);

			Debug.Log($"Rebuilt BVH ({bvhNodeBuffer.Length} nodes for {entityBuffer.Length} entities)");
		}
#endif // BVH

#if BVH_ITERATIVE
		public bool HitWorld(Ray r, out HitRecord hitRec)
		{
			BvhNode** nodes = stackalloc BvhNode*[bvhNodeBuffer.Length];
			Entity* entities = stackalloc Entity[entityBuffer.Length];
#if BVH_SIMD
			int maxVectorWorkingSizePerEntity = sizeof(Sphere4) / sizeof(float4);
			var entityGroupCount = (int) ceil(entityBuffer.Length / 4.0f);
			float4* vectors = stackalloc float4[maxVectorWorkingSizePerEntity * entityGroupCount];
#endif
			var workingArea = new AccumulateJob.WorkingArea
			{
				Nodes = nodes,
				Entities = entities,
#if BVH_SIMD
				Vectors = vectors
#endif
			};

#if FULL_DIAGNOSTICS
			Diagnostics _ = default;
			return World->Hit(r, 0, float.PositiveInfinity, workingArea, ref _, out hitRec);
#else
			return World->Hit(r, 0, float.PositiveInfinity, workingArea, out hitRec);
#endif
		}

#else // !BVH_ITERATIVE
		public bool HitWorld(Ray r, out HitRecord hitRec)
		{
#if BVH_RECURSIVE
			return World->Hit(r, 0, float.PositiveInfinity, out hitRec);
#else
			return World.Hit(r, 0, float.PositiveInfinity, out hitRec);
#endif
		}
#endif

		void CollectActiveEntities()
		{
			activeEntities.Clear();

			if (!scene || scene.Entities == null) return;

			foreach (EntityData entity in scene.Entities)
				if (entity.Enabled)
					activeEntities.Add(entity);

			var rng = new Random(scene.RandomSeed);
			foreach (RandomSphereGroup group in scene.RandomSphereGroups)
			{
				MaterialData GetMaterial()
				{
					(float lambertian, float metal, float dielectric) probabilities = (
						group.LambertChance,
						group.MetalChance,
						group.DieletricChance);

					float sum = probabilities.lambertian + probabilities.metal + probabilities.dielectric;
					probabilities.metal += probabilities.lambertian;
					probabilities.dielectric += probabilities.metal;
					probabilities.lambertian /= sum;
					probabilities.metal /= sum;
					probabilities.dielectric /= sum;

					MaterialData material = null;
					float randomValue = rng.NextFloat();
					if (randomValue < probabilities.lambertian)
					{
						Color from = group.DiffuseAlbedo.colorKeys[0].color;
						Color to = group.DiffuseAlbedo.colorKeys[1].color;
						float3 color = rng.NextFloat3(from.ToFloat3(), to.ToFloat3());
						if (group.DoubleSampleDiffuseAlbedo)
							color *= rng.NextFloat3(from.ToFloat3(), to.ToFloat3());
						material = MaterialData.Lambertian(TextureData.Constant(color), 1);
					}
					else if (randomValue < probabilities.metal)
					{
						Color from = group.MetalAlbedo.colorKeys[0].color;
						Color to = group.MetalAlbedo.colorKeys[1].color;
						float3 color = rng.NextFloat3(from.ToFloat3(), to.ToFloat3());
						float fuzz = rng.NextFloat(group.Fuzz.x, group.Fuzz.y);
						material = MaterialData.Metal(TextureData.Constant(color), 1, fuzz);
					}
					else if (randomValue < probabilities.dielectric)
					{
						material = MaterialData.Dielectric(rng.NextFloat(
							group.RefractiveIndex.x,
							group.RefractiveIndex.y));
					}

					return material;
				}

				bool AnyOverlap(float3 center, float radius) => activeEntities.Where(x => x.Type == EntityType.Sphere)
					.Any(x => !x.SphereData.ExcludeFromOverlapTest &&
					          distance(x.Position, center) < x.SphereData.Radius + radius + group.MinDistance);

				EntityData GetSphere(float3 center, float radius)
				{
					bool moving = rng.NextFloat() < group.MovementChance;
					if (moving)
					{
						float3 offset = rng.NextFloat3(
							float3(group.MovementXOffset.x, group.MovementYOffset.x, group.MovementZOffset.x),
							float3(group.MovementXOffset.y, group.MovementYOffset.y, group.MovementZOffset.y));

						// TODO: reimplement moving spheres
						//return EntityData.Sphere(new SphereData(center, offset, 0, 1, radius), GetMaterial());
					}
					return EntityData.Sphere(center, radius, GetMaterial());
				}

				switch (group.Distribution)
				{
					case RandomDistribution.DartThrowing:
						for (int i = 0; i < group.TentativeCount; i++)
						{
							float3 center = rng.NextFloat3(
								float3(group.CenterX.x, group.CenterY.x, group.CenterZ.x),
								float3(group.CenterX.y, group.CenterY.y, group.CenterZ.y));

							float radius = rng.NextFloat(group.Radius.x, group.Radius.y);

							if (AnyOverlap(center, radius))
								continue;

							activeEntities.Add(GetSphere(center, radius));
						}
						break;

					case RandomDistribution.JitteredGrid:
						float3 ranges = float3(
							group.CenterX.y - group.CenterX.x,
							group.CenterY.y - group.CenterY.x,
							group.CenterZ.y - group.CenterZ.x);

						float3 cellSize = float3(group.PeriodX, group.PeriodY, group.PeriodZ) * sign(ranges);

						// correct the range so that it produces the same result as the book
						float3 correctedRangeEnd = float3(group.CenterX.y, group.CenterY.y, group.CenterZ.y);
						float3 period = max(float3(group.PeriodX, group.PeriodY, group.PeriodZ), 1);
						correctedRangeEnd += (1 - abs(sign(ranges))) * period / 2;

						for (float i = group.CenterX.x; i < correctedRangeEnd.x; i += period.x)
						for (float j = group.CenterY.x; j < correctedRangeEnd.y; j += period.y)
						for (float k = group.CenterZ.x; k < correctedRangeEnd.z; k += period.z)
						{
							float3 center = float3(i, j, k) + rng.NextFloat3(group.Variation * cellSize);
							float radius = rng.NextFloat(group.Radius.x, group.Radius.y);

							if (AnyOverlap(center, radius))
								continue;

							activeEntities.Add(GetSphere(center, radius));
						}
						break;
				}
			}

			Debug.Log($"Collected {activeEntities.Count} active entities");
		}
	}
}