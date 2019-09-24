using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AOT;
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
	public enum DenoiseMode
	{
		None,
		OpenImageDenoise,
		NvidiaOptix
	}

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
		[SerializeField] DenoiseMode denoiseMode = DenoiseMode.None;
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

		Pool<NativeArray<float4>> float4Buffers;
		Pool<NativeArray<float3>> float3Buffers;

		NativeArray<float4> colorAccumulationBuffer;
		NativeArray<float3> normalAccumulationBuffer, albedoAccumulationBuffer;

		Texture2D frontBufferTexture, normalsTexture, albedoTexture, diagnosticsTexture;
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

		OpenImageDenoise.NativeApi.Device denoiseDevice;
		OpenImageDenoise.NativeApi.Filter denoiseFilter;

		struct ActiveJobData<T>
		{
			public JobHandle Handle;
			public Action OnComplete;
			public T OutputData;

			public void Complete()
			{
				Handle.Complete();
				OnComplete?.Invoke();
			}
		}
		struct ActiveJobData
		{
			public JobHandle Handle;
			public Action OnComplete;

			public void Complete()
			{
				Handle.Complete();
				OnComplete?.Invoke();
			}
		}

		struct AccumulateOutputData
		{
			public NativeArray<float4> Color;
			public NativeArray<float3> Normal, Albedo;
		}
		struct PassOutputData
		{
			public NativeArray<float3> Color, Normal, Albedo;
		}

		readonly Queue<ActiveJobData<AccumulateOutputData>> activeAccumulateJobs = new Queue<ActiveJobData<AccumulateOutputData>>();
		readonly Queue<ActiveJobData<PassOutputData>> activeCombineJobs = new Queue<ActiveJobData<PassOutputData>>();
		readonly Queue<ActiveJobData<PassOutputData>> activeDenoiseJobs = new Queue<ActiveJobData<PassOutputData>>();
		ActiveJobData? activeFinalizeJob;

		bool commandBufferHooked, worldNeedsRebuild, initialized, traceAborted, ignoreBatchTimings;
		float focusDistance;
		int lastTraceDepth;
		uint lastSamplesPerPixel;
		ImportanceSamplingMode lastSamplingMode;
		DenoiseMode lastDenoise;

		readonly Stopwatch batchTimer = new Stopwatch(), traceTimer = new Stopwatch();
		readonly List<float> mraysPerSecResults = new List<float>();

		internal readonly List<EntityData> ActiveEntities = new List<EntityData>();
		readonly List<MaterialData> activeMaterials = new List<MaterialData>();

		float2 bufferSize;

		bool TraceActive => activeAccumulateJobs.Count > 0 || activeCombineJobs.Count > 0 ||
		                    activeDenoiseJobs.Count > 0 || activeFinalizeJob.HasValue;

		int BufferLength => (int) (bufferSize.x * bufferSize.y);

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

			float3Buffers = new Pool<NativeArray<float3>>(() =>
					new NativeArray<float3>(BufferLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory),
				itemNameOverride: "NativeArray<float3>") { Capacity = 64 };

			float4Buffers = new Pool<NativeArray<float4>>(() =>
					new NativeArray<float4>(BufferLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory),
				itemNameOverride: "NativeArray<float4>") { Capacity = 16 };

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
			denoiseDevice = OpenImageDenoise.NativeApi.Device.New(OpenImageDenoise.NativeApi.Device.Type.Default);
			OpenImageDenoise.NativeApi.Device.SetErrorFunction(denoiseDevice, OnOidnError, IntPtr.Zero);
			OpenImageDenoise.NativeApi.Device.Commit(denoiseDevice);

            denoiseFilter = OpenImageDenoise.NativeApi.Filter.New(denoiseDevice, "RT");
            OpenImageDenoise.NativeApi.Filter.Set(denoiseFilter, "hdr", true);

            // test for OptiX
            var optixDeviceContext = OptiX.NativeApi.DeviceContext.Create(OnOptixError, 4);
            if (optixDeviceContext.Handle != IntPtr.Zero) Debug.Log("Successfully created OptiX Device Context!");
            var optixDenoiseOptions = new OptiX.NativeApi.Denoiser.Options
            {
	            InputKind = OptiX.NativeApi.Denoiser.InputKind.RgbAlbedoNormal,
	            PixelFormat = OptiX.NativeApi.OptixPixelFormat.Half3
            };
            OptiX.NativeApi.Denoiser denoiser = default;
            var status = OptiX.NativeApi.Denoiser.Create(optixDeviceContext, &optixDenoiseOptions, ref denoiser);
            if (status == OptiX.NativeApi.OptixResult.Success) Debug.Log("Successfully created OptiX Denoiser!");
            status = OptiX.NativeApi.Denoiser.Destroy(denoiser);
            if (status == OptiX.NativeApi.OptixResult.Success) Debug.Log("Successfully destroyed OptiX Denoiser!");
            OptiX.NativeApi.DeviceContext.Destroy(optixDeviceContext);
		}

		[MonoPInvokeCallback(typeof(OpenImageDenoise.NativeApi.ErrorFunction))]
		static void OnOidnError(IntPtr userPtr, OpenImageDenoise.NativeApi.Error code, string message)
		{
			if (string.IsNullOrWhiteSpace(message))
				Debug.LogError(code);
			else
				Debug.LogError($"{code} : {message}");
		}

		[MonoPInvokeCallback(typeof(OpenImageDenoise.NativeApi.ErrorFunction))]
		static void OnOptixError(uint level, string tag, string message, IntPtr cbdata)
		{
			switch (level)
			{
				case 1: Debug.LogError($"nVidia OptiX Fatal Error : {tag} - {message}"); break;
				case 2: Debug.LogError($"nVidia OptiX Error : {tag} - {message}"); break;
				case 3: Debug.LogWarning($"nVidia OptiX Warning : {tag} - {message}"); break;
				case 4: Debug.Log($"nVidia OptiX Trace : {tag} - {message}"); break;
			}
		}

		void OnDestroy()
		{
			// if there is any running job, wait for completion
			// TODO: cancellation
			foreach (var jobData in activeAccumulateJobs) jobData.Complete();
			foreach (var jobData in activeCombineJobs) jobData.Complete();
			foreach (var jobData in activeDenoiseJobs) jobData.Complete();
			activeFinalizeJob?.Complete();

			entityBuffer.SafeDispose();
			importanceSamplingEntityBuffer.SafeDispose();
			sphereBuffer.SafeDispose();
			rectBuffer.SafeDispose();
			boxBuffer.SafeDispose();

			float3Buffers.ReturnAll();
			float3Buffers.Capacity = 0;

			float4Buffers.ReturnAll();
			float4Buffers.Capacity = 0;

#if BVH
			bvhNodeBuffer.SafeDispose();
			bvhNodeMetadataBuffer.SafeDispose();
#endif
			perlinData.Dispose();
#if PATH_DEBUGGING
			debugPaths.SafeDispose();
#endif

			OpenImageDenoise.NativeApi.Filter.Release(denoiseFilter);
			OpenImageDenoise.NativeApi.Device.Release(denoiseDevice);

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
			bool denoiseChanged = lastDenoise != denoiseMode;

			bool traceNeedsReset = buffersNeedRebuild || worldNeedsRebuild || cameraDirty || traceDepthChanged ||
			                       samplingModeChanged || samplesPerPixelDecreased || denoiseChanged;
			bool traceNeedsKick = traceNeedsReset || !stopWhenCompleted;

			if (activeAccumulateJobs.Count > 0 && activeAccumulateJobs.Peek().Handle.IsCompleted)
			{
				ActiveJobData<AccumulateOutputData> completedJob = activeAccumulateJobs.Dequeue();
				completedJob.Complete();

				TimeSpan elapsedTime = batchTimer.Elapsed;
				float totalRayCount = diagnosticsBuffer.Sum(x => x.RayCount);

				AccumulatedSamples += samplesPerBatch;
				LastBatchDuration = (float) elapsedTime.TotalMilliseconds;
				MillionRaysPerSecond = totalRayCount / (float) elapsedTime.TotalSeconds / 1000000;
				if (!ignoreBatchTimings) mraysPerSecResults.Add(MillionRaysPerSecond);
				AvgMRaysPerSecond = mraysPerSecResults.Count == 0 ? 0 : mraysPerSecResults.Average();
				ignoreBatchTimings = false;

				if (traceAborted)
				{
					float4Buffers.Return(completedJob.OutputData.Color);
					float3Buffers.Return(completedJob.OutputData.Normal);
					float3Buffers.Return(completedJob.OutputData.Albedo);
				}
				else
				{
					if (AccumulatedSamples >= samplesPerPixel || previewAfterBatch)
						ScheduleCombine(completedJob.OutputData);

					if (AccumulatedSamples < samplesPerPixel && !traceNeedsReset)
						ScheduleAccumulate(AccumulatedSamples >= samplesPerPixel);
				}
			}

			if (activeCombineJobs.Count > 0 && activeCombineJobs.Peek().Handle.IsCompleted &&
			    (denoiseMode != DenoiseMode.None || !activeFinalizeJob.HasValue))
			{
				ActiveJobData<PassOutputData> completedJob = activeCombineJobs.Dequeue();
				completedJob.Complete();

				if (traceAborted)
				{
					float3Buffers.Return(completedJob.OutputData.Color);
					float3Buffers.Return(completedJob.OutputData.Normal);
					float3Buffers.Return(completedJob.OutputData.Albedo);
				}
				else
				{
					if (denoiseMode != DenoiseMode.None)
						ScheduleDenoise(completedJob.OutputData);
					else
						ScheduleFinalize(completedJob.OutputData);
				}
			}

			if (activeDenoiseJobs.Count > 0 && activeDenoiseJobs.Peek().Handle.IsCompleted &&
			    (!activeFinalizeJob.HasValue || traceAborted))
			{
				ActiveJobData<PassOutputData> completedJob = activeDenoiseJobs.Dequeue();
				completedJob.Complete();

				if (!traceAborted)
					ScheduleFinalize(completedJob.OutputData);
			}

			if (activeFinalizeJob.HasValue && activeFinalizeJob.Value.Handle.IsCompleted)
			{
				activeFinalizeJob.Value.Complete();
				activeFinalizeJob = null;

				if (AccumulatedSamples >= samplesPerPixel)
					LastTraceDuration = (float) traceTimer.Elapsed.TotalMilliseconds;

				SwapBuffers();
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
			}

			if (!TraceActive && traceNeedsKick)
			{
				if (buffersNeedRebuild) EnsureBuffersBuilt();
				if (worldNeedsRebuild) RebuildWorld();
				if (cameraDirty) CleanCamera();

				ScheduleAccumulate(traceNeedsReset || AccumulatedSamples >= samplesPerPixel);
			}

			// TODO: is this needed? when?
			if (!TraceActive && !commandBufferHooked)
				SwapBuffers();
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
				if (!colorAccumulationBuffer.IsCreated) colorAccumulationBuffer = float4Buffers.Take();
				if (!normalAccumulationBuffer.IsCreated) normalAccumulationBuffer = float3Buffers.Take();
				if (!albedoAccumulationBuffer.IsCreated) albedoAccumulationBuffer = float3Buffers.Take();

				colorAccumulationBuffer.ZeroMemory();
				normalAccumulationBuffer.ZeroMemory();
				albedoAccumulationBuffer.ZeroMemory();

#if PATH_DEBUGGING
				debugPaths.EnsureCapacity((int) traceDepth);
#endif
				mraysPerSecResults.Clear();
				AccumulatedSamples = 0;
				lastTraceDepth = traceDepth;
				lastSamplingMode = importanceSampling;
				lastSamplesPerPixel = samplesPerPixel;
				lastDenoise = denoiseMode;
				traceAborted = false;
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
			}

			NativeArray<float4> colorOutputBuffer = float4Buffers.Take();
			NativeArray<float3> normalOutputBuffer = float3Buffers.Take();
			NativeArray<float3> albedoOutputBuffer = float3Buffers.Take();

			var accumulateJob = new AccumulateJob
			{
				InputColor = colorAccumulationBuffer,
				InputNormal = normalAccumulationBuffer,
				InputAlbedo = albedoAccumulationBuffer,

				OutputColor = colorOutputBuffer,
				OutputNormal = normalOutputBuffer,
				OutputAlbedo = albedoOutputBuffer,

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
				DebugCoordinates = int2 (bufferSize / 2)
#endif
			};

			JobHandle accumulateJobHandle = accumulateJob.Schedule(totalBufferSize, 1);

			var copyOutputData = new AccumulateOutputData
			{
				Color = float4Buffers.Take(),
				Normal = float3Buffers.Take(),
				Albedo = float3Buffers.Take()
			};

			JobHandle combinedDependency = JobHandle.CombineDependencies(
				new CopyFloat4BufferJob { Input = colorOutputBuffer, Output = copyOutputData.Color }.Schedule(accumulateJobHandle),
				new CopyFloat3BufferJob { Input = normalOutputBuffer, Output = copyOutputData.Normal }.Schedule(accumulateJobHandle),
				new CopyFloat3BufferJob { Input = albedoOutputBuffer, Output = copyOutputData.Albedo }.Schedule(accumulateJobHandle));

			NativeArray<float4> colorInputBuffer = colorAccumulationBuffer;
			NativeArray<float3> normalInputBuffer = normalAccumulationBuffer,
				albedoInputBuffer = albedoAccumulationBuffer;

			activeAccumulateJobs.Enqueue(new ActiveJobData<AccumulateOutputData>
			{
				Handle = combinedDependency,
				OutputData = copyOutputData,
				OnComplete = () =>
				{
					float4Buffers.Return(colorInputBuffer);
					float3Buffers.Return(normalInputBuffer);
					float3Buffers.Return(albedoInputBuffer);
				}
			});

			// cycle accumulation output into the next accumulation pass's input
			colorAccumulationBuffer = colorOutputBuffer;
			normalAccumulationBuffer = normalOutputBuffer;
			albedoAccumulationBuffer = albedoOutputBuffer;

			batchTimer.Restart();
			if (firstBatch) traceTimer.Restart();
		}

		void ScheduleCombine(AccumulateOutputData accumulateOutput)
		{
			// Debug.Log("Scheduling combine");

			var combineJob = new CombineJob
			{
				InputColor = accumulateOutput.Color,
				InputNormal = accumulateOutput.Normal,
				InputAlbedo = accumulateOutput.Albedo,

				OutputColor = float3Buffers.Take(),
				OutputNormal = float3Buffers.Take(),
				OutputAlbedo = float3Buffers.Take()
			};

			var totalBufferSize = (int) (bufferSize.x * bufferSize.y);
			JobHandle combineJobHandle = combineJob.Schedule(totalBufferSize, 128);

			var copyOutputData = new PassOutputData
			{
				Color = float3Buffers.Take(),
				// reuse what we can!
				Normal = accumulateOutput.Normal,
				Albedo = accumulateOutput.Albedo
			};

			JobHandle combinedDependency = JobHandle.CombineDependencies(
				new CopyFloat3BufferJob { Input = combineJob.OutputColor, Output = copyOutputData.Color }.Schedule(combineJobHandle),
				new CopyFloat3BufferJob { Input = combineJob.OutputNormal, Output = copyOutputData.Normal }.Schedule(combineJobHandle),
				new CopyFloat3BufferJob { Input = combineJob.OutputAlbedo, Output = copyOutputData.Albedo }.Schedule(combineJobHandle));

			activeCombineJobs.Enqueue(new ActiveJobData<PassOutputData>
			{
				Handle = combinedDependency,
				OutputData = copyOutputData,
				OnComplete = () =>
				{
					float4Buffers.Return(accumulateOutput.Color);
					float3Buffers.Return(combineJob.OutputColor);
					float3Buffers.Return(combineJob.OutputNormal);
					float3Buffers.Return(combineJob.OutputAlbedo);
				}
			});
		}

		void ScheduleDenoise(PassOutputData combineOutput)
		{
			// Debug.Log("Scheduling denoise");

			int width = (int) bufferSize.x, height = (int) bufferSize.y;

			NativeArray<float3> denoiseColorOutputBuffer = float3Buffers.Take();

			OpenImageDenoise.NativeApi.Filter.SetSharedImage(denoiseFilter, "color", new IntPtr(combineOutput.Color.GetUnsafePtr()),
				OpenImageDenoise.NativeApi.Buffer.Format.Float3, (ulong) width, (ulong) height, 0, 0, 0);
			OpenImageDenoise.NativeApi.Filter.SetSharedImage(denoiseFilter, "normal", new IntPtr(combineOutput.Normal.GetUnsafePtr()),
				OpenImageDenoise.NativeApi.Buffer.Format.Float3, (ulong) width, (ulong) height, 0, 0, 0);
			OpenImageDenoise.NativeApi.Filter.SetSharedImage(denoiseFilter, "albedo", new IntPtr(combineOutput.Albedo.GetUnsafePtr()),
				OpenImageDenoise.NativeApi.Buffer.Format.Float3, (ulong) width, (ulong) height, 0, 0, 0);
			OpenImageDenoise.NativeApi.Filter.SetSharedImage(denoiseFilter, "output", new IntPtr(denoiseColorOutputBuffer.GetUnsafePtr()),
				OpenImageDenoise.NativeApi.Buffer.Format.Float3, (ulong) width, (ulong) height, 0, 0, 0);
			OpenImageDenoise.NativeApi.Filter.Commit(denoiseFilter);

			var denoiseJob = new DenoiseJob { DenoiseFilter = denoiseFilter };
			JobHandle denoiseJobHandle = denoiseJob.Schedule();

			var copyOutputData = new PassOutputData
			{
				Color = combineOutput.Color,
				Normal = combineOutput.Normal,
				Albedo = combineOutput.Albedo
			};

			JobHandle copyJobHandle = new CopyFloat3BufferJob
				{ Input = denoiseColorOutputBuffer, Output = copyOutputData.Color }.Schedule(denoiseJobHandle);

			activeDenoiseJobs.Enqueue(new ActiveJobData<PassOutputData>
			{
				Handle = copyJobHandle,
				OutputData = copyOutputData,
				OnComplete = () => float3Buffers.Return(denoiseColorOutputBuffer)
			});
		}

		void ScheduleFinalize(PassOutputData lastPassOutput)
		{
			// Debug.Log("Scheduling finalize");

			var finalizeJob = new FinalizeTexturesJob
			{
				InputColor = lastPassOutput.Color,
				InputNormal = lastPassOutput.Normal,
				InputAlbedo = lastPassOutput.Albedo,

				OutputColor = frontBuffer,
				OutputNormal = normalsBuffer,
				OutputAlbedo = albedoBuffer
			};

			var totalBufferSize = (int) (bufferSize.x * bufferSize.y);
			JobHandle finalizeJobHandle = finalizeJob.Schedule(totalBufferSize, 128);

			activeFinalizeJob = new ActiveJobData
			{
				Handle = finalizeJobHandle,
				OnComplete = () =>
				{
					float3Buffers.Return(lastPassOutput.Color);
					float3Buffers.Return(lastPassOutput.Normal);
					float3Buffers.Return(lastPassOutput.Albedo);
				}
			};
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

			float2 lastBufferSize = bufferSize;
			bufferSize = float2(width, height);

			if ((int) lastBufferSize.x != width || (int) lastBufferSize.y != height)
			{
				float3Buffers.Reset();
				float4Buffers.Reset();
				colorAccumulationBuffer = default;
				albedoAccumulationBuffer = normalAccumulationBuffer = default;
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

				Debug.Log($"Rebuilt textures (now {width} x {height})");
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