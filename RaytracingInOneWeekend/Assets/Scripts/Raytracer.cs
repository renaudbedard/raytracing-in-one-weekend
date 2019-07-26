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
using UnityEngine.Rendering;
using static Unity.Mathematics.math;
using Debug = UnityEngine.Debug;
using float3 = Unity.Mathematics.float3;
using Random = Unity.Mathematics.Random;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using Title = UnityEngine.HeaderAttribute;
#endif

namespace RaytracerInOneWeekend
{
	partial class Raytracer : MonoBehaviour
	{
		[Title("References")]
		[SerializeField] UnityEngine.Camera targetCamera = null;

		[Title("Settings")]
		[SerializeField] [Range(0.01f, 2)] float resolutionScaling = 0.5f;
		[SerializeField] [Range(1, 2000)] uint samplesPerPixel = 2000;
		[SerializeField] [Range(1, 100)] uint samplesPerBatch = 10;
		[SerializeField] [Range(1, 100)] uint traceDepth = 35;
		[SerializeField] bool previewAfterBatch = true;
		[SerializeField] bool stopWhenCompleted = true;

		[Title("Camera")]
		[SerializeField] float cameraAperture = 0.1f;

		[Title("World")]
#if ODIN_INSPECTOR
		[InlineEditor(DrawHeader = false)]
#endif
		[SerializeField] SceneData scene = null;

		[Title("Debug")]
		[UsedImplicitly]
#if ODIN_INSPECTOR
		[ShowInInspector]
		[Sirenix.OdinInspector.ReadOnly]
#else
		public
#endif
		uint accumulatedSamples;

		[UsedImplicitly]
#if ODIN_INSPECTOR
		[ShowInInspector]
		[Sirenix.OdinInspector.ReadOnly]
#else
		public
#endif
		float millionRaysPerSecond, avgMRaysPerSecond, lastBatchDuration, lastTraceDuration;

		UnityEngine.Material viewRangeMaterial;
		Texture2D frontBufferTexture, diagnosticsTexture;
		NativeArray<RGBA32> frontBuffer;
		NativeArray<Diagnostics> diagnosticsBuffer;

		// TODO: allocation order of all these buffers is probably pretty crucial for cache locality

		CommandBuffer commandBuffer;
		NativeArray<float4> accumulationInputBuffer, accumulationOutputBuffer;

#if BUFFERED_MATERIALS
		NativeArray<Material> materialBuffer;
#endif

#if SOA_SIMD
		SoaSpheres sphereBuffer;
		internal SoaSpheres World => sphereBuffer;
#elif AOSOA_SIMD
		AosoaSpheres sphereBuffer;
		internal AosoaSpheres World => sphereBuffer;
#else // !SOA_SIMD && !AOSOA_SIMD
		NativeArray<Sphere> sphereBuffer;
		NativeArray<Entity> entityBuffer;
#if !BVH
		NativeArray<Entity> World => entityBuffer;
#endif
#endif

#if BVH
		NativeList<BvhNode> bvhNodeBuffer;
		BvhNode World => bvhNodeBuffer.IsCreated ? bvhNodeBuffer[bvhNodeBuffer.Length - 1] : default;
#endif

#if BVH_ITERATIVE
		NativeArray<IntPtr> nodeWorkingBuffer;
		NativeArray<Entity> entityWorkingBuffer;
#endif
#if BVH_SIMD
		NativeArray<float4> vectorWorkingBuffer;
#endif

		JobHandle? accumulateJobHandle;
		JobHandle? combineJobHandle;

		bool commandBufferHooked;
		bool worldNeedsRebuild;
		float lastFieldOfView;
		bool initialized;
		float focusDistance;
		bool traceAborted;
		bool ignoreBatchTimings;

		readonly Stopwatch batchTimer = new Stopwatch();
		readonly Stopwatch traceTimer = new Stopwatch();
		readonly List<float> mraysPerSecResults = new List<float>();

		readonly List<SphereData> activeSpheres = new List<SphereData>();
		readonly List<MaterialData> activeMaterials = new List<MaterialData>();

		float2 bufferSize;

		bool TraceActive => accumulateJobHandle.HasValue || combineJobHandle.HasValue;

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

			viewRangeMaterial = new UnityEngine.Material(Shader.Find("Hidden/ViewRange"));

			ignoreBatchTimings = true;
		}

		void Start()
		{
			targetCamera.RemoveAllCommandBuffers();

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
			if (entityBuffer.IsCreated) entityBuffer.Dispose();
			if (sphereBuffer.IsCreated) sphereBuffer.Dispose();
#endif
#if BUFFERED_MATERIALS
			if (materialBuffer.IsCreated) materialBuffer.Dispose();
#endif
			if (accumulationInputBuffer.IsCreated) accumulationInputBuffer.Dispose();
			if (accumulationOutputBuffer.IsCreated) accumulationOutputBuffer.Dispose();
#if BVH
			if (bvhNodeBuffer.IsCreated) bvhNodeBuffer.Dispose();
#endif
#if BVH_ITERATIVE
			if (nodeWorkingBuffer.IsCreated) nodeWorkingBuffer.Dispose();
			if (entityWorkingBuffer.IsCreated) entityWorkingBuffer.Dispose();
#endif
#if BVH_SIMD
			if (vectorWorkingBuffer.IsCreated) vectorWorkingBuffer.Dispose();
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
			bool cameraDirty = targetCamera.transform.hasChanged ||
							   !Mathf.Approximately(lastFieldOfView, targetCamera.fieldOfView);
			bool traceNeedsReset = buffersNeedRebuild || worldNeedsRebuild || cameraDirty;

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

				accumulatedSamples += samplesPerBatch;
				lastBatchDuration = (float) elapsedTime.TotalMilliseconds;
				millionRaysPerSecond = totalRayCount / (float) elapsedTime.TotalSeconds / 1000000;
				if (!ignoreBatchTimings) mraysPerSecResults.Add(millionRaysPerSecond);
				avgMRaysPerSecond = mraysPerSecResults.Count == 0 ? 0 : mraysPerSecResults.Average();
				ignoreBatchTimings = false;
			}

			if (!TraceActive && traceNeedsReset)
			{
				RebuildDirtyComponents();
				ScheduleAccumulate(true);
			}

			if (combineJobHandle.HasValue && combineJobHandle.Value.IsCompleted)
			{
				if (accumulateJobHandle.HasValue && accumulateJobHandle.Value.IsCompleted)
					CompleteAccumulate();

				combineJobHandle.Value.Complete();
				combineJobHandle = null;

				bool traceCompleted = false;
				if (accumulatedSamples >= samplesPerPixel)
				{
					traceCompleted = true;
					lastTraceDuration = (float) traceTimer.Elapsed.TotalMilliseconds;
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
			lastFieldOfView = targetCamera.fieldOfView;
			targetCamera.transform.hasChanged = false;
		}

		void SwapBuffers()
		{
			float bufferMin = float.MaxValue, bufferMax = float.MinValue;

			switch (bufferView)
			{
				case BufferView.RayCount:
					viewRangeMaterial.SetInt("_Channel", 0);
					foreach (Diagnostics value in diagnosticsBuffer)
					{
						bufferMin = min(bufferMin, value.RayCount);
						bufferMax = max(bufferMax, value.RayCount);
					}
					break;

#if FULL_DIAGNOSTICS && BVH_ITERATIVE
				case BufferView.BvhHitCount:
					viewRangeMaterial.SetInt("_Channel", 1);
					foreach (Diagnostics value in diagnosticsBuffer)
					{
						bufferMin = min(bufferMin, value.BoundsHitCount);
						bufferMax = max(bufferMax, value.BoundsHitCount);
					}
					break;

				case BufferView.CandidateCount:
					viewRangeMaterial.SetInt("_Channel", 2);
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
					diagnosticsTexture.Apply(false);
					viewRangeMaterial.SetFloat("_Minimum", bufferMin);
					viewRangeMaterial.SetFloat("_Range", bufferMax - bufferMin);
					break;
			}

			if (!commandBufferHooked)
			{
				commandBuffer.Clear();

				var blitTarget = new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget);
				switch (bufferView)
				{
					case BufferView.Front: commandBuffer.Blit(frontBufferTexture, blitTarget); break;
					default: commandBuffer.Blit(diagnosticsTexture, blitTarget, viewRangeMaterial); break;
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

			var raytracingCamera = new Camera(origin, lookAt, cameraTransform.up, targetCamera.fieldOfView,
				bufferSize.x / bufferSize.y, cameraAperture, focusDistance);

			var totalBufferSize = (int) (bufferSize.x * bufferSize.y);

			if (firstBatch)
			{
				if (accumulationInputBuffer.IsCreated) accumulationInputBuffer.Dispose();
				accumulationInputBuffer = new NativeArray<float4>(totalBufferSize, Allocator.Persistent);

				mraysPerSecResults.Clear();
				accumulatedSamples = 0;
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
				OutputSamples = accumulationOutputBuffer,
				OutputDiagnostics = diagnosticsBuffer,
#if BUFFERED_MATERIALS
				Material = materialBuffer,
#endif
#if BVH_ITERATIVE
				ThreadCount = SystemInfo.processorCount,
				NodeWorkingBuffer = nodeWorkingBuffer,
				EntityWorkingBuffer = entityWorkingBuffer,
#endif
#if BVH_SIMD
				VectorWorkingBuffer = vectorWorkingBuffer,
#endif
			};

			accumulateJobHandle = accumulateJob.Schedule(totalBufferSize, 1);

			if (accumulatedSamples + samplesPerBatch >= samplesPerPixel || previewAfterBatch)
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

			if (accumulationOutputBuffer.Length != width * height)
			{
				if (accumulationOutputBuffer.IsCreated) accumulationOutputBuffer.Dispose();
				accumulationOutputBuffer = new NativeArray<float4>(width * height,
					Allocator.Persistent,
					NativeArrayOptions.UninitializedMemory);

				Debug.Log($"Rebuilt accumulation output buffer (now {width} x {height})");
			}

			bufferSize = float2(width, height);
		}

		void RebuildWorld()
		{
#if UNITY_EDITOR
			if (scene) scene.ClearDirty();
#endif
			CollectActiveSpheres();

			activeMaterials.Clear();
			foreach (SphereData sphere in activeSpheres)
				if (!activeMaterials.Contains(sphere.Material))
					activeMaterials.Add(sphere.Material);

#if BUFFERED_MATERIALS
			int materialCount = activeMaterials.Count;
			if (materialBuffer.Length != materialCount)
			{
				if (materialBuffer.IsCreated) materialBuffer.Dispose();
				materialBuffer = new NativeArray<Material>(materialCount, Allocator.Persistent);
			}

			for (var i = 0; i < activeMaterials.Count; i++)
			{
				MaterialData material = activeMaterials[i];
				TextureData albedo = material.Albedo;
				materialBuffer[i] = new Material(material.Type,
					albedo
						? new Texture(albedo.Type, albedo.MainColor.ToFloat3(), albedo.SecondaryColor.ToFloat3())
						: default,
					material.Fuzz, material.RefractiveIndex);
			}
#endif

#if SOA_SIMD || AOSOA_SIMD
			int sphereCount = activeSpheres.Count;
			if (sphereBuffer.Length != sphereCount)
			{
				sphereBuffer.Dispose();
#if SOA_SIMD
				sphereBuffer = new SoaSpheres(sphereCount);
#elif AOSOA_SIMD
				sphereBuffer = new AosoaSpheres(sphereCount);
#endif
			}

			for (var i = 0; i < activeSpheres.Count; i++)
			{
				SphereData sphereData = activeSpheres[i];
				sphereBuffer.SetElement(i, sphereData.Center, sphereData.Radius);

				MaterialData material = sphereData.Material;
#if BUFFERED_MATERIALS
				sphereBuffer.MaterialIndex[i] = activeMaterials.IndexOf(material);
#else
				TextureData albedo = material.Albedo;
				sphereBuffer.Material[i] =
					new Material(material.Type,
						albedo
							? new Texture(albedo.Type, albedo.MainColor.ToFloat3(), albedo.SecondaryColor.ToFloat3())
							: default,
						material.Fuzz, material.RefractiveIndex);
#endif
			}

#else // !SOA_SIMD && !AOSOA_SIMD
			RebuildEntityBuffer();
#endif

#if BVH
			RebuildBvh();
#endif

			worldNeedsRebuild = false;

			Debug.Log($"Rebuilt world ({activeSpheres.Count} spheres, {activeMaterials.Count} materials)");
		}

#if !SOA_SIMD && !AOSOA_SIMD
		void RebuildEntityBuffer()
		{
			int entityCount = activeSpheres.Count;

			if (!entityBuffer.IsCreated || entityBuffer.Length != entityCount)
			{
				if (entityBuffer.IsCreated) entityBuffer.Dispose();
				entityBuffer = new NativeArray<Entity>(entityCount, Allocator.Persistent);
			}

			if (!sphereBuffer.IsCreated || sphereBuffer.Length != activeSpheres.Count)
			{
				if (sphereBuffer.IsCreated) sphereBuffer.Dispose();
				sphereBuffer = new NativeArray<Sphere>(activeSpheres.Count, Allocator.Persistent);
			}

			int entityIndex = 0;

			for (var i = 0; i < activeSpheres.Count; i++)
			{
				SphereData sphereData = activeSpheres[i];
				MaterialData material = sphereData.Material;
#if !BUFFERED_MATERIALS
				TextureData texture = material ? material.Albedo : null;
#endif
				sphereBuffer[i] = new Sphere(sphereData.Center, sphereData.Radius,
#if BUFFERED_MATERIALS
					activeMaterials.IndexOf(material));
#else
					new Material(material.Type, texture
							? new Texture(texture.Type, texture.MainColor.ToFloat3(), texture.SecondaryColor.ToFloat3())
							: default,
						material.Fuzz, material.RefractiveIndex));
#endif
				unsafe
				{
					entityBuffer[entityIndex++] = new Entity((Sphere*) sphereBuffer.GetUnsafePtr() + i);
				}
			}
		}
#endif // !SOA_SIMD && !AOSOA_SIMD

#if BVH
		void RebuildBvh()
		{
			// TODO: figure out how many nodes we need for a given entity count
			if (!bvhNodeBuffer.IsCreated) bvhNodeBuffer = new NativeList<BvhNode>(1024, Allocator.Persistent);
			bvhNodeBuffer.Clear();

			bvhNodeBuffer.Add(new BvhNode(entityBuffer, bvhNodeBuffer));

#if BVH_ITERATIVE
			int workingBufferSize = entityBuffer.Length * SystemInfo.processorCount;
			if (!nodeWorkingBuffer.IsCreated || nodeWorkingBuffer.Length != workingBufferSize)
			{
				if (nodeWorkingBuffer.IsCreated) nodeWorkingBuffer.Dispose();
				nodeWorkingBuffer = new NativeArray<IntPtr>(workingBufferSize, Allocator.Persistent);
			}
			if (!entityWorkingBuffer.IsCreated || entityWorkingBuffer.Length != workingBufferSize)
			{
				if (entityWorkingBuffer.IsCreated) entityWorkingBuffer.Dispose();
				entityWorkingBuffer = new NativeArray<Entity>(workingBufferSize, Allocator.Persistent);
			}
#endif
#if BVH_SIMD
			if (!vectorWorkingBuffer.IsCreated || vectorWorkingBuffer.Length != workingBufferSize * Sphere4.StreamCount)
			{
				if (vectorWorkingBuffer.IsCreated) vectorWorkingBuffer.Dispose();
				vectorWorkingBuffer = new NativeArray<float4>(workingBufferSize * Sphere4.StreamCount, Allocator.Persistent);
			}
#endif
			Debug.Log($"Rebuilt BVH ({bvhNodeBuffer.Length} nodes for {entityBuffer.Length} entities)");
		}
#endif // BVH

#if BVH_ITERATIVE
		public unsafe bool HitWorld(Ray r, out HitRecord hitRec)
		{
			var workingArea = new AccumulateJob.WorkingArea
			{
				Nodes = (BvhNode**) nodeWorkingBuffer.GetUnsafeReadOnlyPtr(),
				Entities = (Entity*) entityWorkingBuffer.GetUnsafeReadOnlyPtr(),
#if BVH_SIMD
				Vectors = (float4*) vectorWorkingBuffer.GetUnsafeReadOnlyPtr()
#endif
			};
#if FULL_DIAGNOSTICS && BVH_ITERATIVE
			Diagnostics _ = default;
			return World.Hit(r, 0, float.PositiveInfinity, workingArea, ref _, out hitRec);
#else
			return World.Hit(r, 0, float.PositiveInfinity, workingArea, out hitRec);
#endif
		}

#else // !BVH_ITERATIVE
		public bool HitWorld(Ray r, out HitRecord hitRec)
		{
			return World.Hit(r, 0, float.PositiveInfinity, out hitRec);
		}
#endif

		void CollectActiveSpheres()
		{
			activeSpheres.Clear();

			if (!scene) return;

			foreach (SphereData sphere in scene.Spheres)
				if (sphere.Enabled)
					activeSpheres.Add(sphere);

			Random rng = new Random(scene.RandomSeed);
			foreach (RandomSphereGroup group in scene.RandomSphereGroups)
			{
				MaterialData GetMaterial()
				{
					(float lambertian, float metal, float dielectric) probabilities = (
						group.LambertianProbability,
						group.MetalProbability,
						group.DieletricProbability);

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
						Color from = group.AlbedoColor.colorKeys[0].color;
						Color to = group.AlbedoColor.colorKeys[1].color;
						float3 color = rng.NextFloat3(from.ToFloat3(), to.ToFloat3());
						material = MaterialData.Lambertian(TextureData.Constant(color));
					}
					else if (randomValue < probabilities.metal)
					{
						Color from = group.AlbedoColor.colorKeys[0].color;
						Color to = group.AlbedoColor.colorKeys[1].color;
						float3 color = rng.NextFloat3(from.ToFloat3(), to.ToFloat3());
						float fuzz = rng.NextFloat(group.Fuzz.x, group.Fuzz.y);
						material = MaterialData.Metal(TextureData.Constant(color), fuzz);
					}
					else if (randomValue < probabilities.dielectric)
					{
						material = MaterialData.Dielectric(rng.NextFloat(
							group.RefractiveIndex.x,
							group.RefractiveIndex.y));
					}

					return material;
				}

				switch (group.Distribution)
				{
					case RandomDistribution.WhiteNoise:
						for (int i = 0; i < group.Count; i++)
						{
							float3 center = rng.NextFloat3(
								float3(group.CenterX.x, group.CenterY.x, group.CenterZ.x),
								float3(group.CenterX.y, group.CenterY.y, group.CenterZ.y));

							float radius = rng.NextFloat(group.Radius.x, group.Radius.y);

							activeSpheres.Add(new SphereData(center, radius, GetMaterial()));
						}
						break;

					case RandomDistribution.JitteredGrid:
						float3 ranges = float3(
							group.CenterX.y - group.CenterX.x,
							group.CenterY.y - group.CenterY.x,
							group.CenterZ.y - group.CenterZ.x);

						float3 cellSize = float3(group.PeriodX, group.PeriodY, group.PeriodZ) * sign(ranges);

						for (float i = group.CenterX.x; i <= group.CenterX.y; i += max(group.PeriodX, 1))
						for (float j = group.CenterY.x; j <= group.CenterY.y; j += max(group.PeriodY, 1))
						for (float k = group.CenterZ.x; k <= group.CenterZ.y; k += max(group.PeriodZ, 1))
						{
							float3 center = float3(i, j, k) + rng.NextFloat3(group.Variation * cellSize);
							float radius = rng.NextFloat(group.Radius.x, group.Radius.y);

							activeSpheres.Add(new SphereData(center, radius, GetMaterial()));
						}
						break;
				}
			}
		}

		void BuildRandomScene()
		{
			activeSpheres.Clear();

			var rng = new Random(scene.RandomSeed);

			for (int a = -11; a < 11; a++)
			{
				for (int b = -11; b < 11; b++)
				{
					float materialProb = rng.NextFloat();
					float3 center = float3(a + 0.9f * rng.NextFloat(), 0.2f, b + 0.9f * rng.NextFloat());

					if (distance(center, float3(4, 0.2f, 0)) <= 0.9)
						continue;

					if (materialProb < 0.8)
					{
						activeSpheres.Add(new SphereData(center, 0.2f,
							MaterialData.Lambertian(TextureData.Constant(rng.NextFloat3() * rng.NextFloat3()))));
					}
					else if (materialProb < 0.95)
					{
						activeSpheres.Add(new SphereData(center, 0.2f,
							MaterialData.Metal(TextureData.Constant(rng.NextFloat3(0.5f, 1)), rng.NextFloat(0, 0.5f))));
					}
					else
						activeSpheres.Add(new SphereData(center, 0.2f, MaterialData.Dielectric(1.5f)));
				}
			}
		}
	}
}