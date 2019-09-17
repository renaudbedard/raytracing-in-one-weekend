using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AOT;
using JetBrains.Annotations;
using OpenImageDenoise;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
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
		[SerializeField] [Range(1, 500)] int traceDepth = 35;
		[SerializeField] ImportanceSamplingMode importanceSampling = ImportanceSamplingMode.None;
		[SerializeField] bool denoise = true;
		[SerializeField] bool subPixelJitter = true;
		[SerializeField] bool previewAfterBatch = true;
		[SerializeField] bool stopWhenCompleted = true;
#if PATH_DEBUGGING
		[SerializeField] bool fadeDebugPaths = false;
		[SerializeField] [Range(0, 25)] float debugPathDuration = 1;
#endif

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

		// accumulation
		NativeArray<float4> accumulateColorInputBuffer, accumulateColorOutputBuffer;
		NativeArray<float3> accumulateNormalInputBuffer, accumulateNormalOutputBuffer;
		NativeArray<float3> accumulateAlbedoInputBuffer, accumulateAlbedoOutputBuffer;

		// combination
		NativeArray<float4> combineColorInputBuffer;
		NativeArray<float3> combineNormalInputBuffer, combineAlbedoInputBuffer;
		NativeArray<float3> combineColorOutputBuffer, combineNormalOutputBuffer, combineAlbedoOutputBuffer;

		// denoising
		NativeArray<float3> denoiseColorInputBuffer, denoiseNormalInputBuffer, denoiseAlbedoInputBuffer;
		NativeArray<float3> denoiseOutputBuffer;

		// finalization
		Texture2D frontBufferTexture, normalsTexture, albedoTexture, diagnosticsTexture;
		NativeArray<float3> finalizeColorInputBuffer, finalizeNormalInputBuffer, finalizeAlbedoInputBuffer;
		NativeArray<RGBA32> frontBuffer, normalsBuffer, albedoBuffer;

		CommandBuffer commandBuffer;
		UnityEngine.Material viewRangeMaterial;
		NativeArray<Diagnostics> diagnosticsBuffer;

		NativeArray<Sphere> sphereBuffer;
		NativeArray<Rect> rectBuffer;
		NativeArray<Box> boxBuffer;
		NativeArray<Entity> entityBuffer, importanceSamplingEntityBuffer;
#if !BVH
		NativeArray<Entity> World => entityBuffer;
#endif
#if PATH_DEBUGGING
		NativeArray<DebugPath> debugPaths;
#endif

#if BVH
		NativeList<BvhNode> bvhNodeBuffer;
		NativeList<BvhNodeMetadata> bvhNodeMetadataBuffer;
		BvhNode* BvhRoot => bvhNodeBuffer.IsCreated ? (BvhNode*) bvhNodeBuffer.GetUnsafePtr() : null;
#endif

		readonly PerlinDataGenerator perlinData = new PerlinDataGenerator();

		Denoise.Device denoiseDevice;
		Denoise.Filter denoiseFilter;

		JobHandle? accumulateJobHandle, combineJobHandle, denoiseJobHandle, finalizeJobHandle;

		bool commandBufferHooked, worldNeedsRebuild, initialized, traceAborted, ignoreBatchTimings;
		float focusDistance;
		int lastTraceDepth;
		uint lastSamplesPerPixel;
		ImportanceSamplingMode lastSamplingMode;
		bool lastDenoise;

		readonly Stopwatch batchTimer = new Stopwatch(), traceTimer = new Stopwatch();
		readonly List<float> mraysPerSecResults = new List<float>();

		internal readonly List<EntityData> ActiveEntities = new List<EntityData>();
		readonly List<MaterialData> activeMaterials = new List<MaterialData>();

		float2 bufferSize;

		bool TraceActive => accumulateJobHandle.HasValue || combineJobHandle.HasValue || denoiseJobHandle.HasValue ||
		                    finalizeJobHandle.HasValue;

		enum BufferView
		{
			Front,
			RayCount,
			Normals,
			Albedo,
#if FULL_DIAGNOSTICS && BVH_ITERATIVE
			BvhHitCount,
			CandidateCount
#endif
		}

#if !UNITY_EDITOR
		const BufferView bufferView = BufferView.Front;
#endif
		int channelPropertyId, minimumRangePropertyId, normalDisplayId;

		void Awake()
		{
			commandBuffer = new CommandBuffer { name = "Raytracer" };

			const HideFlags flags = HideFlags.HideAndDontSave;
			frontBufferTexture = new Texture2D(0, 0, TextureFormat.RGBA32, false) { hideFlags = flags };
			normalsTexture = new Texture2D(0, 0, TextureFormat.RGBA32, false) { hideFlags = flags };
			albedoTexture = new Texture2D(0, 0, TextureFormat.RGBA32, false) { hideFlags = flags };
#if FULL_DIAGNOSTICS && BVH_ITERATIVE
			diagnosticsTexture = new Texture2D(0, 0, TextureFormat.RGBAFloat, false) { hideFlags = flags };
#else
			diagnosticsTexture = new Texture2D(0, 0, TextureFormat.RFloat, false) { hideFlags = flags };
#endif

			viewRangeMaterial = new UnityEngine.Material(viewRangeShader);
			channelPropertyId = Shader.PropertyToID("_Channel");
			minimumRangePropertyId = Shader.PropertyToID("_Minimum_Range");
			normalDisplayId = Shader.PropertyToID("_NormalDisplay");

			ignoreBatchTimings = true;
		}

		void Start()
		{
			targetCamera.RemoveAllCommandBuffers();
#if UNITY_EDITOR
			scene = scene.DeepClone();
#endif
			RebuildWorld();
			InitDenoiser();
			EnsureBuffersBuilt();
			CleanCamera();

			ScheduleAccumulate(true);
		}

		void InitDenoiser()
		{
			denoiseDevice = Denoise.Device.New(Denoise.Device.Type.Default);
			Denoise.Device.SetErrorFunction(denoiseDevice, OnDenoiseError, IntPtr.Zero);
            Denoise.Device.Commit(denoiseDevice);

            denoiseFilter = Denoise.Filter.New(denoiseDevice, "RT");
            Denoise.Filter.Set(denoiseFilter, "hdr", true);
		}

		[MonoPInvokeCallback(typeof(Denoise.ErrorFunction))]
		static void OnDenoiseError(IntPtr userPtr, Denoise.Error code, string message)
		{
			if (string.IsNullOrWhiteSpace(message))
				Debug.LogError(code);
			else
				Debug.LogError($"{code} : {message}");
		}

		void OnDestroy()
		{
			// if there is a running job, wait for completion
			// TODO: cancellation
			accumulateJobHandle?.Complete();
			combineJobHandle?.Complete();
			denoiseJobHandle?.Complete();
			finalizeJobHandle?.Complete();

			entityBuffer.SafeDispose();
			importanceSamplingEntityBuffer.SafeDispose();
			sphereBuffer.SafeDispose();
			rectBuffer.SafeDispose();
			boxBuffer.SafeDispose();

			// accumulate
			accumulateColorInputBuffer.SafeDispose();
			accumulateNormalInputBuffer.SafeDispose();
			accumulateAlbedoInputBuffer.SafeDispose();
			accumulateColorOutputBuffer.SafeDispose();
			accumulateNormalOutputBuffer.SafeDispose();
			accumulateAlbedoOutputBuffer.SafeDispose();

			// combine
			combineColorInputBuffer.SafeDispose();
			combineNormalInputBuffer.SafeDispose();
			combineAlbedoInputBuffer.SafeDispose();
			combineColorOutputBuffer.SafeDispose();
			combineNormalOutputBuffer.SafeDispose();
			combineAlbedoOutputBuffer.SafeDispose();

			// denoise
			denoiseColorInputBuffer.SafeDispose();
			denoiseNormalInputBuffer.SafeDispose();
			denoiseAlbedoInputBuffer.SafeDispose();
			denoiseOutputBuffer.SafeDispose();

			// finalize
			finalizeColorInputBuffer.SafeDispose();
			finalizeNormalInputBuffer.SafeDispose();
			finalizeAlbedoInputBuffer.SafeDispose();

#if BVH
			bvhNodeBuffer.SafeDispose();
			bvhNodeMetadataBuffer.SafeDispose();
#endif
			perlinData.Dispose();
#if PATH_DEBUGGING
			debugPaths.SafeDispose();
#endif

			Denoise.Filter.Release(denoiseFilter);
			Denoise.Device.Release(denoiseDevice);

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
			bool samplingModeChanged = importanceSampling != lastSamplingMode;
			bool samplesPerPixelDecreased = lastSamplesPerPixel != samplesPerPixel && AccumulatedSamples > samplesPerPixel;
			bool denoiseChanged = lastDenoise != denoise;

			bool traceNeedsReset = buffersNeedRebuild || worldNeedsRebuild || cameraDirty || traceDepthChanged ||
			                       samplingModeChanged || samplesPerPixelDecreased || denoiseChanged;
			bool traceNeedsKick = traceNeedsReset; // TODO

			void RebuildDirtyComponents()
			{
				if (buffersNeedRebuild) EnsureBuffersBuilt();
				if (worldNeedsRebuild) RebuildWorld();
				if (cameraDirty) CleanCamera();
			}

			if (accumulateJobHandle.HasValue && accumulateJobHandle.Value.IsCompleted)
			{
				accumulateJobHandle.Value.Complete();
				accumulateJobHandle = null;

				TimeSpan elapsedTime = batchTimer.Elapsed;
				float totalRayCount = diagnosticsBuffer.Sum(x => x.RayCount);

				AccumulatedSamples += samplesPerBatch;
				LastBatchDuration = (float) elapsedTime.TotalMilliseconds;
				MillionRaysPerSecond = totalRayCount / (float) elapsedTime.TotalSeconds / 1000000;
				if (!ignoreBatchTimings) mraysPerSecResults.Add(MillionRaysPerSecond);
				AvgMRaysPerSecond = mraysPerSecResults.Count == 0 ? 0 : mraysPerSecResults.Average();
				ignoreBatchTimings = false;

				if (traceAborted) { traceAborted = false; return; }

				if (AccumulatedSamples >= samplesPerPixel || previewAfterBatch)
					ScheduleCombine();

				if (AccumulatedSamples < samplesPerPixel || !stopWhenCompleted)
					ScheduleAccumulate(AccumulatedSamples >= samplesPerPixel);
			}

			if (combineJobHandle.HasValue && combineJobHandle.Value.IsCompleted)
			{
				combineJobHandle.Value.Complete();
				combineJobHandle = null;

				if (traceAborted) { traceAborted = false; return; }

				if (denoise) ScheduleDenoise();
				else ScheduleFinalize();
			}

			if (denoiseJobHandle.HasValue && denoiseJobHandle.Value.IsCompleted)
			{
				denoiseJobHandle.Value.Complete();
				denoiseJobHandle = null;

				if (traceAborted) { traceAborted = false; return; }

				ScheduleFinalize();
			}

			if (finalizeJobHandle.HasValue && finalizeJobHandle.Value.IsCompleted)
			{
				finalizeJobHandle.Value.Complete();
				finalizeJobHandle = null;

				if (AccumulatedSamples >= samplesPerPixel)
					LastTraceDuration = (float) traceTimer.Elapsed.TotalMilliseconds;

				SwapBuffers();
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
				RebuildDirtyComponents();

				traceAborted = false;
			}

			if (!TraceActive && traceNeedsKick)
			{
				RebuildDirtyComponents();
				ScheduleAccumulate(traceNeedsReset);
			}

			if (!TraceActive && !commandBufferHooked)
				SwapBuffers();

			lastSamplesPerPixel = samplesPerPixel;
			lastDenoise = denoise;
		}

		void ScheduleAccumulate(bool firstBatch)
		{
			// Debug.Log($"Scheduling accumulate (firstBatch = {firstBatch})");

			Transform cameraTransform = targetCamera.transform;
			Vector3 origin = cameraTransform.localPosition;
			Vector3 lookAt = origin + cameraTransform.forward;

			if (HitWorld(new Ray(origin, cameraTransform.forward), out HitRecord hitRec))
				focusDistance = hitRec.Distance;

			var raytracingCamera = new Camera(origin, lookAt, cameraTransform.up, scene.CameraFieldOfView,
				bufferSize.x / bufferSize.y, scene.CameraAperture, focusDistance);

			var totalBufferSize = (int) (bufferSize.x * bufferSize.y);

			if (firstBatch)
			{
				accumulateColorInputBuffer.ZeroMemory();
				accumulateNormalInputBuffer.ZeroMemory();
				accumulateAlbedoInputBuffer.ZeroMemory();

#if PATH_DEBUGGING
				debugPaths.EnsureCapacity((int) traceDepth);
#endif
				mraysPerSecResults.Clear();
				AccumulatedSamples = 0;
				lastTraceDepth = traceDepth;
				lastSamplingMode = importanceSampling;
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
			}
			else
			{
				Util.Swap(ref accumulateColorInputBuffer, ref accumulateColorOutputBuffer);
				Util.Swap(ref accumulateNormalInputBuffer, ref accumulateNormalOutputBuffer);
				Util.Swap(ref accumulateAlbedoInputBuffer, ref accumulateAlbedoOutputBuffer);
			}

			var accumulateJob = new AccumulateJob
			{
				InputColor = accumulateColorInputBuffer,
				InputNormal = accumulateNormalInputBuffer,
				InputAlbedo = accumulateAlbedoInputBuffer,

				OutputColor = accumulateColorOutputBuffer,
				OutputNormal = accumulateNormalOutputBuffer,
				OutputAlbedo = accumulateAlbedoOutputBuffer,

				Size = bufferSize,
				Camera = raytracingCamera,
				SkyBottomColor = scene.SkyBottomColor.ToFloat3(),
				SkyTopColor = scene.SkyTopColor.ToFloat3(),
				Seed = (uint) Time.frameCount + 1,
				SampleCount = min(samplesPerPixel, samplesPerBatch),
				TraceDepth = traceDepth,
				SubPixelJitter = subPixelJitter,
				Entities = entityBuffer,
#if BVH
				BvhRoot = BvhRoot,
#endif
				PerlinData = perlinData.GetRuntimeData(),
				OutputDiagnostics = diagnosticsBuffer,
				ImportanceSampler = new ImportanceSampler
				{
					TargetEntities = importanceSamplingEntityBuffer,
					Mode = importanceSampling
				},
#if BVH_ITERATIVE
				NodeCount = bvhNodeBuffer.Length,
#endif
#if PATH_DEBUGGING
				DebugPaths = (DebugPath*) debugPaths.GetUnsafePtr(),
				DebugCoordinates = int2(bufferSize / 2)
#endif
			};

			accumulateJobHandle = accumulateJob.Schedule(totalBufferSize, 1);

			JobHandle copyDependencies = accumulateJobHandle.Value;
			if (combineJobHandle.HasValue)
				copyDependencies = JobHandle.CombineDependencies(copyDependencies, combineJobHandle.Value);

			accumulateJobHandle = JobHandle.CombineDependencies(accumulateJobHandle.Value,
				new CopyFloat4BufferJob { Input = accumulateColorOutputBuffer, Output = combineColorInputBuffer }
					.Schedule(copyDependencies));
			accumulateJobHandle = JobHandle.CombineDependencies(accumulateJobHandle.Value,
				new CopyFloat3BufferJob { Input = accumulateNormalOutputBuffer, Output = combineNormalInputBuffer }
					.Schedule(copyDependencies));
			accumulateJobHandle = JobHandle.CombineDependencies(accumulateJobHandle.Value,
					new CopyFloat3BufferJob { Input = accumulateAlbedoOutputBuffer, Output = combineAlbedoInputBuffer }
					.Schedule(copyDependencies));

			batchTimer.Restart();
			if (firstBatch) traceTimer.Restart();
		}

		void ScheduleCombine()
		{
			// Debug.Log("Scheduling combine");

			var combineJob = new CombineJob
			{
				InputColor = combineColorInputBuffer,
				InputNormal = combineNormalInputBuffer,
				InputAlbedo = combineAlbedoInputBuffer,

				OutputColor = combineColorOutputBuffer,
				OutputNormal = combineNormalOutputBuffer,
				OutputAlbedo = combineAlbedoOutputBuffer
			};

			var length = (int) (bufferSize.x * bufferSize.y);
			combineJobHandle = combineJob.Schedule(length, 128);

			JobHandle copyDependencies = combineJobHandle.Value;
			if (denoise && denoiseJobHandle.HasValue)
				copyDependencies = JobHandle.CombineDependencies(copyDependencies, denoiseJobHandle.Value);
			else if (!denoise && finalizeJobHandle.HasValue)
				copyDependencies = JobHandle.CombineDependencies(copyDependencies, finalizeJobHandle.Value);

			var copyJobs = new []
			{
				new CopyFloat3BufferJob { Input = combineColorOutputBuffer, Output = denoise ? denoiseColorInputBuffer : finalizeColorInputBuffer },
				new CopyFloat3BufferJob { Input = combineNormalOutputBuffer, Output = denoise ? denoiseNormalInputBuffer : finalizeNormalInputBuffer },
				new CopyFloat3BufferJob { Input = combineAlbedoOutputBuffer, Output = denoise ? denoiseAlbedoInputBuffer : finalizeAlbedoInputBuffer }
			};
			foreach (CopyFloat3BufferJob copyJob in copyJobs)
				combineJobHandle = JobHandle.CombineDependencies(combineJobHandle.Value, copyJob.Schedule(copyDependencies));
		}

		void ScheduleDenoise()
		{
			// Debug.Log("Scheduling denoise");

			var denoiseJob = new DenoiseJob { DenoiseFilter = denoiseFilter };
			denoiseJobHandle = denoiseJob.Schedule();

			JobHandle copyDependencies = denoiseJobHandle.Value;
			if (finalizeJobHandle.HasValue)
				copyDependencies = JobHandle.CombineDependencies(copyDependencies, finalizeJobHandle.Value);

			var copyJobs = new []
			{
				new CopyFloat3BufferJob { Input = denoiseOutputBuffer, Output = finalizeColorInputBuffer },
				new CopyFloat3BufferJob { Input = denoiseNormalInputBuffer, Output = finalizeNormalInputBuffer },
				new CopyFloat3BufferJob { Input = denoiseAlbedoInputBuffer, Output = finalizeAlbedoInputBuffer }
			};
			foreach (CopyFloat3BufferJob copyJob in copyJobs)
				denoiseJobHandle = JobHandle.CombineDependencies(denoiseJobHandle.Value, copyJob.Schedule(copyDependencies));
		}

		void ScheduleFinalize()
		{
			// Debug.Log("Scheduling finalize");

			var finalizeJob = new FinalizeTexturesJob
			{
				InputColor = finalizeColorInputBuffer,
				InputNormal = finalizeNormalInputBuffer,
				InputAlbedo = finalizeAlbedoInputBuffer,

				OutputColor = frontBuffer,
				OutputNormal = normalsBuffer,
				OutputAlbedo = albedoBuffer
			};
			finalizeJobHandle = finalizeJob.Schedule((int) (bufferSize.x * bufferSize.y), 128);
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
				case BufferView.Normals: normalsTexture.Apply(false); break;
				case BufferView.Albedo: albedoTexture.Apply(false); break;
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
					case BufferView.Front: commandBuffer.Blit(frontBufferTexture, blitTarget); break;
					case BufferView.Normals: commandBuffer.Blit(normalsTexture, blitTarget); break;
					case BufferView.Albedo: commandBuffer.Blit(albedoTexture, blitTarget); break;

					default:
						viewRangeMaterial.SetInt(channelPropertyId, (int) bufferView - 1);
						commandBuffer.Blit(diagnosticsTexture, blitTarget, viewRangeMaterial);
						break;
				}

				targetCamera.AddCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
				commandBufferHooked = true;
			}
		}

		void EnsureBuffersBuilt()
		{
			int width = (int) ceil(targetCamera.pixelWidth * resolutionScaling);
			int height = (int) ceil(targetCamera.pixelHeight * resolutionScaling);
			bufferSize = float2(width, height);

			int bufferLength = width * height;

			if (accumulateColorInputBuffer.EnsureCapacity(bufferLength) |
				accumulateNormalInputBuffer.EnsureCapacity(bufferLength) |
				accumulateAlbedoInputBuffer.EnsureCapacity(bufferLength) |
				accumulateColorOutputBuffer.EnsureCapacity(bufferLength) |
				accumulateNormalOutputBuffer.EnsureCapacity(bufferLength) |
				accumulateAlbedoOutputBuffer.EnsureCapacity(bufferLength))
			{
				Debug.Log($"Rebuilt accumulation buffers (now {width} x {height})");
			}

			if (combineColorInputBuffer.EnsureCapacity(bufferLength) |
			    combineNormalInputBuffer.EnsureCapacity(bufferLength) |
			    combineAlbedoInputBuffer.EnsureCapacity(bufferLength) |
				combineColorOutputBuffer.EnsureCapacity(bufferLength) |
				combineNormalOutputBuffer.EnsureCapacity(bufferLength) |
				combineAlbedoOutputBuffer.EnsureCapacity(bufferLength))
			{
				Debug.Log($"Rebuilt combining buffers (now {width} x {height})");
			}

			if (denoiseColorInputBuffer.EnsureCapacity(bufferLength) |
			    denoiseNormalInputBuffer.EnsureCapacity(bufferLength) |
			    denoiseAlbedoInputBuffer.EnsureCapacity(bufferLength) |
			    denoiseOutputBuffer.EnsureCapacity(bufferLength))
			{
				Debug.Log($"Rebuilt denoising buffers (now {width} x {height})");
			}

			if (finalizeColorInputBuffer.EnsureCapacity(bufferLength) |
			    finalizeNormalInputBuffer.EnsureCapacity(bufferLength) |
			    finalizeAlbedoInputBuffer.EnsureCapacity(bufferLength))
			{
				Debug.Log($"Rebuilt finalization buffers (now {width} x {height})");
			}

			if (frontBufferTexture.width != width || frontBufferTexture.height != height ||
			    diagnosticsTexture.width != width || diagnosticsTexture.height != height ||
			    normalsTexture.width != width || normalsTexture.height != height ||
			    albedoTexture.width != width || albedoTexture.height != height)
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
				PrepareTexture(normalsTexture, out normalsBuffer);
				PrepareTexture(albedoTexture, out albedoBuffer);

				Denoise.Filter.SetSharedImage(denoiseFilter, "color", new IntPtr(denoiseColorInputBuffer.GetUnsafePtr()),
					Denoise.Buffer.Format.Float3, (ulong) width, (ulong) height, 0, 0, 0);
				// Denoise.Filter.SetSharedImage(denoiseFilter, "normal", new IntPtr(denoiseNormalInputBuffer.GetUnsafePtr()),
				// 	Denoise.Buffer.Format.Float3, (ulong) width, (ulong) height, 0, 0, 0);
				// Denoise.Filter.SetSharedImage(denoiseFilter, "albedo", new IntPtr(denoiseAlbedoInputBuffer.GetUnsafePtr()),
				// 	Denoise.Buffer.Format.Float3, (ulong) width, (ulong) height, 0, 0, 0);
				Denoise.Filter.SetSharedImage(denoiseFilter, "output", new IntPtr(denoiseOutputBuffer.GetUnsafePtr()),
					Denoise.Buffer.Format.Float3, (ulong) width, (ulong) height, 0, 0, 0);
				Denoise.Filter.Commit(denoiseFilter);

				Debug.Log($"Rebuilt texture & associated buffers (now {width} x {height})");
			}
		}

		void RebuildWorld()
		{
#if UNITY_EDITOR
			if (scene) scene.ClearDirty();
#endif
			CollectActiveEntities();

			activeMaterials.Clear();
			foreach (EntityData entity in ActiveEntities)
				if (!activeMaterials.Contains(entity.Material))
					activeMaterials.Add(entity.Material);

			RebuildEntityBuffers();
#if BVH
			RebuildBvh();
#endif

			perlinData.Generate(scene.RandomSeed);

			worldNeedsRebuild = false;
		}

		void RebuildEntityBuffers()
		{
			int entityCount = ActiveEntities.Count;

			entityBuffer.EnsureCapacity(entityCount);

			sphereBuffer.EnsureCapacity(ActiveEntities.Count(x => x.Type == EntityType.Sphere));
			rectBuffer.EnsureCapacity(ActiveEntities.Count(x => x.Type == EntityType.Rect));
			boxBuffer.EnsureCapacity(ActiveEntities.Count(x => x.Type == EntityType.Box));

			// TODO: factor in specular materials
			importanceSamplingEntityBuffer.EnsureCapacity(ActiveEntities.Count(x =>
				x.Material.Type == MaterialType.DiffuseLight));

			int entityIndex = 0, sphereIndex = 0, rectIndex = 0, boxIndex = 0, importanceSamplingIndex = 0;
			foreach (EntityData e in ActiveEntities)
			{
				Vector2 sizeFactor = Vector2.one;
				void* contentPointer = null;
				RigidTransform rigidTransform = new RigidTransform(e.Rotation, e.Position);

				switch (e.Type)
				{
					case EntityType.Sphere:
						sphereBuffer[sphereIndex] = new Sphere(e.SphereData.Radius);
						sizeFactor *= e.SphereData.Radius;
						contentPointer = (Sphere*) sphereBuffer.GetUnsafePtr() + sphereIndex++;
						break;

					case EntityType.Rect:
						rectBuffer[rectIndex] = new Rect(e.RectData.Size);
						sizeFactor *= e.RectData.Size;
						contentPointer = (Rect*) rectBuffer.GetUnsafePtr() + rectIndex++;
						break;

					case EntityType.Box:
						boxBuffer[boxIndex] = new Box(e.BoxData.Size);
						sizeFactor *= e.BoxData.Size;
						contentPointer = (Box*) boxBuffer.GetUnsafePtr() + boxIndex++;
						break;
				}

				MaterialData materialData = e.Material;
				TextureData albedo = materialData ? materialData.Albedo : null;
				TextureData emission = materialData ? materialData.Emission : null;

				Material material = materialData
					? new Material(materialData.Type, materialData.TextureScale * sizeFactor,
						albedo.GetRuntimeData(), emission.GetRuntimeData(),
						materialData.Fuzz, materialData.RefractiveIndex, materialData.Density)
					: default;

				Entity entity = e.Moving
					? new Entity(entityIndex, e.Type, contentPointer, rigidTransform, material, e.DestinationOffset, e.TimeRange)
					: new Entity(entityIndex, e.Type, contentPointer, rigidTransform, material);

				entityBuffer[entityIndex++] = entity;

				if (e.Material.Type == MaterialType.DiffuseLight)
					importanceSamplingEntityBuffer[importanceSamplingIndex++] = entity;
			}
		}

#if BVH
		void RebuildBvh()
		{
			bvhNodeBuffer.EnsureCapacity(entityBuffer.Length * 2);
			bvhNodeMetadataBuffer.EnsureCapacity(entityBuffer.Length * 2);

			bvhNodeBuffer.Clear();
			bvhNodeMetadataBuffer.Clear();

			var rootNode = new BvhNode(entityBuffer, bvhNodeBuffer, bvhNodeMetadataBuffer);
			rootNode.Metadata->Id = bvhNodeBuffer.Length;
			bvhNodeBuffer.AddNoResize(rootNode);

			bvhNodeBuffer.AsArray().Sort(new BvhNodeComparer());
			BvhRoot->SetupPointers(bvhNodeBuffer);

			// re-sort the entity buffer so it can be indexed
			entityBuffer.Sort(new EntityIdComparer());

			Debug.Log($"Rebuilt BVH ({bvhNodeBuffer.Length} nodes for {entityBuffer.Length} entities)");
		}
#endif // BVH

#if BVH_ITERATIVE
		public bool HitWorld(Ray r, out HitRecord hitRec)
		{
			BvhNode** nodes = stackalloc BvhNode*[bvhNodeBuffer.Length];
			Entity* entities = stackalloc Entity[entityBuffer.Length];

			var workingArea = new AccumulateJob.WorkingArea
			{
				Nodes = nodes,
				Entities = entities,
			};

			var rng = new Random(scene.RandomSeed);
#if FULL_DIAGNOSTICS
			Diagnostics _ = default;
			return BvhRoot->Hit(entityBuffer, r, 0, float.PositiveInfinity, ref rng, workingArea, ref _, out hitRec);
#else
			return BvhRoot->Hit(entityBuffer, r, 0, float.PositiveInfinity, ref rng, workingArea, out hitRec);
#endif
		}

#else // !BVH_ITERATIVE
		public bool HitWorld(Ray r, out HitRecord hitRec)
		{
			var rng = new Random(scene.RandomSeed);
#if BVH_RECURSIVE
			return World->Hit(r, 0, float.PositiveInfinity, ref rng, out hitRec);
#else
			return World.Hit(r, 0, float.PositiveInfinity, ref rng, out hitRec);
#endif
		}
#endif

		void CollectActiveEntities()
		{
			ActiveEntities.Clear();

			if (!scene) return;

			if (scene.Entities != null)
			{
				foreach (EntityData entity in scene.Entities)
					if (entity.Enabled)
						ActiveEntities.Add(entity);
			}

			if (scene.RandomEntityGroups != null)
			{
				var rng = new Random(scene.RandomSeed);
				foreach (RandomEntityGroup group in scene.RandomEntityGroups)
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

					// TODO: fix overlap test to account for all entity types
					bool AnyOverlap(float3 center, float radius) => ActiveEntities
						.Where(x => x.Type == EntityType.Sphere)
						.Any(x => !x.SphereData.ExcludeFromOverlapTest &&
						          distance(x.Position, center) < x.SphereData.Radius + radius + group.MinDistance);

					EntityData GetEntity(float3 center, float3 radius)
					{
						bool moving = rng.NextFloat() < group.MovementChance;
						quaternion rotation = quaternion.Euler(group.Rotation);
						var entityData = new EntityData
						{
							Type = group.Type,
							Position = rotate(rotation, center - (float3) group.Offset) + (float3) group.Offset,
							Rotation = rotation,
							Material = GetMaterial()
						};
						switch (group.Type)
						{
							case EntityType.Sphere:
								entityData.SphereData = new SphereData(radius.x);
								break;
							case EntityType.Box:
								entityData.BoxData = new BoxData(radius);
								break;
							case EntityType.Rect:
								entityData.RectData = new RectData(radius.xy);
								break;
						}

						if (moving)
						{
							float3 offset = rng.NextFloat3(
								float3(group.MovementXOffset.x, group.MovementYOffset.x, group.MovementZOffset.x),
								float3(group.MovementXOffset.y, group.MovementYOffset.y, group.MovementZOffset.y));

							entityData.TimeRange = new Vector2(0, 1);
							entityData.Moving = true;
							entityData.DestinationOffset = offset;
						}

						return entityData;
					}

					switch (group.Distribution)
					{
						case RandomDistribution.DartThrowing:
							for (int i = 0; i < group.TentativeCount; i++)
							{
								float3 center = rng.NextFloat3(
									float3(-group.SpreadX / 2, -group.SpreadY / 2, -group.SpreadZ / 2),
									float3(group.SpreadX / 2, group.SpreadY / 2, group.SpreadZ / 2));

								center += (float3) group.Offset;

								float radius = rng.NextFloat(group.Radius.x, group.Radius.y);

								if (!group.SkipOverlapTest && AnyOverlap(center, radius))
									continue;

								ActiveEntities.Add(GetEntity(center, radius));
							}

							break;

						case RandomDistribution.JitteredGrid:
							float3 ranges = float3(group.SpreadX, group.SpreadY, group.SpreadZ);
							float3 cellSize = float3(group.PeriodX, group.PeriodY, group.PeriodZ) * sign(ranges);

							// correct the range so that it produces the same result as the book
							float3 correctedRangeEnd = (float3) group.Offset + ranges / 2;
							float3 period = max(float3(group.PeriodX, group.PeriodY, group.PeriodZ), 1);
							correctedRangeEnd += (1 - abs(sign(ranges))) * period / 2;

							for (float i = group.Offset.x - ranges.x / 2; i < correctedRangeEnd.x; i += period.x)
							for (float j = group.Offset.y - ranges.y / 2; j < correctedRangeEnd.y; j += period.y)
							for (float k = group.Offset.z - ranges.z / 2; k < correctedRangeEnd.z; k += period.z)
							{
								float3 center = float3(i, j, k) + rng.NextFloat3(group.PositionVariation * cellSize);
								float3 radius = rng.NextFloat(group.Radius.x, group.Radius.y) *
								                float3(rng.NextFloat(group.ScaleVariationX.x, group.ScaleVariationX.y),
									                rng.NextFloat(group.ScaleVariationY.x, group.ScaleVariationY.y),
									                rng.NextFloat(group.ScaleVariationZ.x, group.ScaleVariationZ.y));

								if (!group.SkipOverlapTest && AnyOverlap(center, radius.x))
									continue;

								ActiveEntities.Add(GetEntity(center, radius));
							}

							break;
					}
				}
			}

			Debug.Log($"Collected {ActiveEntities.Count} active entities");
		}
	}
}