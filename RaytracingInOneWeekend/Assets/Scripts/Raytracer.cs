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
using OptiX;

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

	partial class Raytracer : MonoBehaviour
	{
		[Title("References")]
		[SerializeField] UnityEngine.Camera targetCamera = null;
		[SerializeField] BlueNoise blueNoise;

		[Title("Settings")] [SerializeField] [Range(1, 100)]
		int interlacing = 2;

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

		[Title("World")] [InlineEditor(DrawHeader = false)] [SerializeField]
		internal SceneData scene = null;

		[Title("Debug")] [SerializeField] Shader viewRangeShader = null;
		[OdinReadOnly] public float AccumulatedSamples;

		[UsedImplicitly] [OdinReadOnly] public float MillionRaysPerSecond,
			AvgMRaysPerSecond,
			LastBatchDuration,
			LastBatchRayCount,
			LastTraceDuration;

		[UsedImplicitly] [OdinReadOnly] public float BufferMinValue, BufferMaxValue;

		[UsedImplicitly] [ShowInInspector] public float AccumulateJobs => scheduledAccumulateJobs.Count;
		[UsedImplicitly] [ShowInInspector] public float CombineJobs => scheduledCombineJobs.Count;
		[UsedImplicitly] [ShowInInspector] public float DenoiseJobs => scheduledDenoiseJobs.Count;
		[UsedImplicitly] [ShowInInspector] public float FinalizeJobs => scheduledFinalizeJobs.Count;

		Pool<NativeArray<float4>> float4Buffers;
		Pool<NativeArray<float3>> float3Buffers;
		Pool<NativeArray<int>> intBuffers;
		Pool<NativeArray<long>> longBuffers;
		Pool<NativeArray<bool>> boolBuffers;

		int interlacingOffsetIndex;
		int[] interlacingOffsets;

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
		unsafe BvhNode* BvhRoot => bvhNodeBuffer.IsCreated ? (BvhNode*) bvhNodeBuffer.GetUnsafePtr() : null;
#endif

		readonly PerlinNoise perlinNoise = new PerlinNoise();

		OidnDevice oidnDevice;
		OidnFilter oidnFilter;

		OptixDeviceContext optixDeviceContext;
		OptixDenoiser optixDenoiser;
		OptixDenoiserSizes optixDenoiserSizes;
		CudaStream cudaStream;

		CudaBuffer optixScratchMemory = default,
			optixDenoiserState = default,
			optixColorBuffer = default,
			optixAlbedoBuffer = default,
			optixOutputBuffer = default;

		struct ScheduledJobData<T>
		{
			public JobHandle Handle;
			public Action OnComplete;
			public T OutputData;
			public NativeArray<bool> CancellationToken;

			public unsafe void Cancel()
			{
				*((bool*) NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(CancellationToken)) = true;
			}

			public void Complete()
			{
				Handle.Complete();
				OnComplete?.Invoke();
				OnComplete = null;
			}
		}

		struct AccumulateOutputData
		{
			public NativeArray<float4> Color;
			public NativeArray<float3> Normal, Albedo;
			public NativeArray<int> ReducedRayCount;
			public JobHandle ReduceRayCountJobHandle;
			public NativeArray<long> Timing;
		}

		struct PassOutputData
		{
			public NativeArray<float3> Color, Normal, Albedo;
		}

		readonly Queue<ScheduledJobData<AccumulateOutputData>> scheduledAccumulateJobs = new Queue<ScheduledJobData<AccumulateOutputData>>();
		readonly Queue<ScheduledJobData<PassOutputData>> scheduledCombineJobs = new Queue<ScheduledJobData<PassOutputData>>();
		readonly Queue<ScheduledJobData<PassOutputData>> scheduledDenoiseJobs = new Queue<ScheduledJobData<PassOutputData>>();
		readonly Queue<ScheduledJobData<PassOutputData>> scheduledFinalizeJobs = new Queue<ScheduledJobData<PassOutputData>>();

		bool commandBufferHooked, worldNeedsRebuild, initialized, traceAborted, ignoreBatchTimings;
		float focusDistance;
		int lastTraceDepth;
		uint lastSamplesPerPixel;
		bool queuedAccumulate;
		ImportanceSamplingMode lastSamplingMode;

		readonly Stopwatch traceTimer = new Stopwatch();
		readonly List<float> mraysPerSecResults = new List<float>();

		internal readonly List<EntityData> ActiveEntities = new List<EntityData>();
		readonly List<MaterialData> activeMaterials = new List<MaterialData>();

		float2 bufferSize;

		int BufferLength => (int) (bufferSize.x * bufferSize.y);

		bool TraceActive => scheduledAccumulateJobs.Count > 0 || scheduledCombineJobs.Count > 0 ||
		                    scheduledDenoiseJobs.Count > 0 || scheduledFinalizeJobs.Count > 0;

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
		int channelPropertyId, minimumRangePropertyId;

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

			float3Buffers = new Pool<NativeArray<float3>>(() =>
					new NativeArray<float3>(BufferLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory),
				itemNameOverride: "NativeArray<float3>") { Capacity = 64 };

			float4Buffers = new Pool<NativeArray<float4>>(() =>
					new NativeArray<float4>(BufferLength, Allocator.Persistent, NativeArrayOptions.UninitializedMemory),
				itemNameOverride: "NativeArray<float4>") { Capacity = 16 };

			intBuffers = new Pool<NativeArray<int>>(() =>
					new NativeArray<int>(1, Allocator.Persistent, NativeArrayOptions.UninitializedMemory),
				itemNameOverride: "NativeArray<int>") { Capacity = 4 };

			longBuffers = new Pool<NativeArray<long>>(() =>
					new NativeArray<long>(2, Allocator.Persistent, NativeArrayOptions.UninitializedMemory),
				itemNameOverride: "NativeArray<long>") { Capacity = 4 };

			boolBuffers = new Pool<NativeArray<bool>>(() => new NativeArray<bool>(1, Allocator.Persistent),
				itemNameOverride: "NativeArray<bool>", cleanupMethod: buffer => buffer[0] = false) { Capacity = 64 };

			ignoreBatchTimings = true;
		}

		void Start()
		{
			targetCamera.RemoveAllCommandBuffers();
#if UNITY_EDITOR
			scene = scene.DeepClone();
#endif
			RebuildWorld();
			InitDenoisers();
			EnsureBuffersBuilt();
			CleanCamera();

			ScheduleAccumulate(true);
		}

		void InitDenoisers()
		{
			// Open Image Denoise

			oidnDevice = OidnDevice.New(OidnDevice.Type.Default);
			OidnDevice.SetErrorFunction(oidnDevice, OnOidnError, IntPtr.Zero);
			OidnDevice.Commit(oidnDevice);

			oidnFilter = OidnFilter.New(oidnDevice, "RT");
			OidnFilter.Set(oidnFilter, "hdr", true);

			// OptiX

			CudaError cudaError;
			if ((cudaError = OptixApi.InitializeCuda()) != CudaError.Success)
			{
				Debug.LogError($"CUDA initialization failed : {cudaError}");
				return;
			}

			OptixResult result;
			if ((result = OptixApi.Initialize()) != OptixResult.Success)
			{
				Debug.LogError($"OptiX initialization failed : {result}");
				return;
			}

			var options = new OptixDeviceContextOptions
			{
				LogCallbackFunction = OnOptixError,
				LogCallbackLevel = OptixLogLevel.Warning
			};

			if ((result = OptixDeviceContext.Create(options, ref optixDeviceContext)) != OptixResult.Success)
			{
				Debug.LogError($"Optix device creation failed : {result}");
				return;
			}

			var denoiseOptions = new OptixDenoiserOptions
			{
				InputKind = OptixDenoiserInputKind.RgbAlbedo,
				PixelFormat = OptixPixelFormat.Float3
			};

			unsafe
			{
				OptixDenoiser.Create(optixDeviceContext, &denoiseOptions, ref optixDenoiser);
			}

			OptixDenoiser.SetModel(optixDenoiser, OptixModelKind.Ldr, IntPtr.Zero, 0);

			if ((cudaError = CudaStream.Create(ref cudaStream)) != CudaError.Success)
				Debug.LogError($"CUDA Stream creation failed : {cudaError}");
		}

		[MonoPInvokeCallback(typeof(OidnErrorFunction))]
		static void OnOidnError(IntPtr userPtr, OidnError code, string message)
		{
			if (string.IsNullOrWhiteSpace(message))
				Debug.LogError(code);
			else
				Debug.LogError($"{code} : {message}");
		}

		[MonoPInvokeCallback(typeof(OptixLogCallback))]
		static void OnOptixError(OptixLogLevel level, string tag, string message, IntPtr cbdata)
		{
			switch (level)
			{
				case OptixLogLevel.Fatal:
					Debug.LogError($"nVidia OptiX Fatal Error : {tag} - {message}");
					break;
				case OptixLogLevel.Error:
					Debug.LogError($"nVidia OptiX Error : {tag} - {message}");
					break;
				case OptixLogLevel.Warning:
					Debug.LogWarning($"nVidia OptiX Warning : {tag} - {message}");
					break;
				case OptixLogLevel.Print:
					Debug.Log($"nVidia OptiX Trace : {tag} - {message}");
					break;
			}
		}

		void OnDestroy()
		{
			foreach (var jobData in scheduledAccumulateJobs) jobData.Cancel();
			foreach (var jobData in scheduledCombineJobs) jobData.Cancel();
			foreach (var jobData in scheduledDenoiseJobs) jobData.Cancel();
			foreach (var jobData in scheduledFinalizeJobs) jobData.Cancel();

			foreach (var jobData in scheduledAccumulateJobs) { jobData.Handle.Complete(); jobData.OutputData.ReduceRayCountJobHandle.Complete(); }
			foreach (var jobData in scheduledCombineJobs) jobData.Handle.Complete();
			foreach (var jobData in scheduledDenoiseJobs) jobData.Handle.Complete();
			foreach (var jobData in scheduledFinalizeJobs) jobData.Handle.Complete();

			entityBuffer.SafeDispose();
			importanceSamplingEntityBuffer.SafeDispose();
			sphereBuffer.SafeDispose();
			rectBuffer.SafeDispose();
			boxBuffer.SafeDispose();

			float3Buffers.ReturnAll();
			float3Buffers.Capacity = 0;

			float4Buffers.ReturnAll();
			float4Buffers.Capacity = 0;

			intBuffers.ReturnAll();
			intBuffers.Capacity = 0;

			longBuffers.ReturnAll();
			longBuffers.Capacity = 0;

			boolBuffers.ReturnAll();
			boolBuffers.Capacity = 0;

#if BVH
			bvhNodeBuffer.SafeDispose();
			bvhNodeMetadataBuffer.SafeDispose();
#endif
			perlinNoise.Dispose();
#if PATH_DEBUGGING
			debugPaths.SafeDispose();
#endif

			OidnFilter.Release(oidnFilter);
			OidnDevice.Release(oidnDevice);

			OptixDenoiser.Destroy(optixDenoiser);
			OptixDeviceContext.Destroy(optixDeviceContext);

			void Check(CudaError cudaError)
			{
				if (cudaError != CudaError.Success)
					Debug.LogError($"CUDA Error : {cudaError}");
			}

			Check(CudaStream.Destroy(cudaStream));

			Check(CudaBuffer.Deallocate(optixDenoiserState));
			Check(CudaBuffer.Deallocate(optixScratchMemory));
			Check(CudaBuffer.Deallocate(optixColorBuffer));
			Check(CudaBuffer.Deallocate(optixAlbedoBuffer));
			Check(CudaBuffer.Deallocate(optixOutputBuffer));

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

			bool traceNeedsReset = buffersNeedRebuild || worldNeedsRebuild || cameraDirty || traceDepthChanged ||
			                       samplingModeChanged || samplesPerPixelDecreased;

			if (traceNeedsReset || traceAborted)
			{
				int i = 0;
				bool ShouldCancel() => i++ > 0 || traceAborted;
				foreach (var jobData in scheduledAccumulateJobs) { if (ShouldCancel()) jobData.Cancel(); } i = 0;
				foreach (var jobData in scheduledCombineJobs) { if (ShouldCancel()) jobData.Cancel(); } i = 0;
				foreach (var jobData in scheduledDenoiseJobs) { if (ShouldCancel()) jobData.Cancel(); } i = 0;
				foreach (var jobData in scheduledFinalizeJobs) { if (ShouldCancel()) jobData.Cancel(); } i = 0;

				foreach (var jobData in scheduledAccumulateJobs) jobData.Handle.Complete();
				foreach (var jobData in scheduledCombineJobs) jobData.Handle.Complete();
				foreach (var jobData in scheduledDenoiseJobs) jobData.Handle.Complete();
				foreach (var jobData in scheduledFinalizeJobs) jobData.Handle.Complete();
			}

			while (scheduledAccumulateJobs.Count > 0 && scheduledAccumulateJobs.Peek().Handle.IsCompleted)
			{
				ScheduledJobData<AccumulateOutputData> completedJob = scheduledAccumulateJobs.Dequeue();
				completedJob.Complete();

				TimeSpan elapsedTime = DateTime.FromFileTimeUtc(completedJob.OutputData.Timing[1]) -
				                       DateTime.FromFileTimeUtc(completedJob.OutputData.Timing[0]);
				longBuffers.Return(completedJob.OutputData.Timing);

				completedJob.OutputData.ReduceRayCountJobHandle.Complete();
				int totalRayCount = completedJob.OutputData.ReducedRayCount[0];
				intBuffers.Return(completedJob.OutputData.ReducedRayCount);

				LastBatchRayCount = totalRayCount;
				AccumulatedSamples += (float) samplesPerBatch / interlacing;
				LastBatchDuration = (float) elapsedTime.TotalMilliseconds;
				MillionRaysPerSecond = totalRayCount / (float) elapsedTime.TotalSeconds / 1000000;
				if (!ignoreBatchTimings) mraysPerSecResults.Add(MillionRaysPerSecond);
				AvgMRaysPerSecond = mraysPerSecResults.Count == 0 ? 0 : mraysPerSecResults.Average();
				ignoreBatchTimings = false;
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif

				queuedAccumulate = !traceAborted && AccumulatedSamples < samplesPerPixel;
			}

			while (scheduledCombineJobs.Count > 0 && scheduledCombineJobs.Peek().Handle.IsCompleted)
			{
				ScheduledJobData<PassOutputData> completedJob = scheduledCombineJobs.Dequeue();
				completedJob.Complete();
			}

			while (scheduledDenoiseJobs.Count > 0 && scheduledDenoiseJobs.Peek().Handle.IsCompleted)
			{
				ScheduledJobData<PassOutputData> completedJob = scheduledDenoiseJobs.Dequeue();
				completedJob.Complete();
			}

			while (scheduledFinalizeJobs.Count > 0 && scheduledFinalizeJobs.Peek().Handle.IsCompleted)
			{
				ScheduledJobData<PassOutputData> completedJob = scheduledFinalizeJobs.Dequeue();
				completedJob.Complete();

				if (AccumulatedSamples >= samplesPerPixel)
					LastTraceDuration = (float) traceTimer.Elapsed.TotalMilliseconds;

				if (!traceAborted)
					SwapBuffers();
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
			}

			if (!TraceActive)
			{
				if (buffersNeedRebuild) EnsureBuffersBuilt();
				if (worldNeedsRebuild) RebuildWorld();
			}
			if (cameraDirty) CleanCamera();

			// kick if needed (with double-buffering)
			if ((queuedAccumulate && scheduledFinalizeJobs.Count <= 1) ||
			    (traceNeedsReset && !traceAborted) ||
			    (!TraceActive && !stopWhenCompleted && !traceAborted))
			{
				ScheduleAccumulate(traceNeedsReset || AccumulatedSamples >= samplesPerPixel,
					scheduledAccumulateJobs.Count > 0
						? (JobHandle?) JobHandle.CombineDependencies(
							scheduledAccumulateJobs.Peek().Handle,
							scheduledAccumulateJobs.Peek().OutputData.ReduceRayCountJobHandle)
						: null);

				queuedAccumulate = false;
			}
		}

		void ScheduleAccumulate(bool firstBatch, JobHandle? dependency = null)
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

				interlacingOffsetIndex = 0;

#if PATH_DEBUGGING
				debugPaths.EnsureCapacity((int) traceDepth);
#endif
				mraysPerSecResults.Clear();
				AccumulatedSamples = 0;
				lastTraceDepth = traceDepth;
				lastSamplingMode = importanceSampling;
				lastSamplesPerPixel = samplesPerPixel;
				traceAborted = false;
#if UNITY_EDITOR
				ForceUpdateInspector();
#endif
			}

			NativeArray<float4> colorOutputBuffer = float4Buffers.Take();
			NativeArray<float3> normalOutputBuffer = float3Buffers.Take();
			NativeArray<float3> albedoOutputBuffer = float3Buffers.Take();
			NativeArray<bool> cancellationBuffer = boolBuffers.Take();

			if (interlacingOffsets == null || interlacing != interlacingOffsets.Length)
				interlacingOffsets = Util.SpaceFillingSeries(interlacing).ToArray();

			if (interlacingOffsetIndex >= interlacing)
				interlacingOffsetIndex = 0;

			AccumulateJob accumulateJob;
			uint frameSeed = (uint) Time.frameCount + 1;

			unsafe
			{
				accumulateJob = new AccumulateJob
				{
					CancellationToken = cancellationBuffer,

					InputColor = colorAccumulationBuffer,
					InputNormal = normalAccumulationBuffer,
					InputAlbedo = albedoAccumulationBuffer,

					OutputColor = colorOutputBuffer,
					OutputNormal = normalOutputBuffer,
					OutputAlbedo = albedoOutputBuffer,

					SliceOffset = interlacingOffsets[interlacingOffsetIndex++],
					SliceDivider = interlacing,

					Size = bufferSize,
					Camera = raytracingCamera,
					Environment = new Environment
					{
						SkyBottomColor = scene.SkyBottomColor.ToFloat3(),
						SkyTopColor = scene.SkyTopColor.ToFloat3(),
						SkyCubemap = scene.SkyCubemap ? new Cubemap(scene.SkyCubemap) : default,
						SkyType = scene.SkyType,
					},
					Seed = frameSeed,
					SampleCount = min(samplesPerPixel, samplesPerBatch),
					TraceDepth = traceDepth,
					SubPixelJitter = subPixelJitter,
					Entities = entityBuffer,
#if BVH
					BvhRoot = BvhRoot,
#endif
					PerlinNoise = perlinNoise.GetRuntimeData(),
					BlueNoise = blueNoise.GetRuntimeData(frameSeed),
					OutputDiagnostics = diagnosticsBuffer,
					ImportanceSampler = new ImportanceSampler
					{
						TargetEntities = importanceSamplingEntityBuffer,
						Mode = importanceSamplingEntityBuffer.Length == 0
							? ImportanceSamplingMode.None
							: importanceSampling
					},
#if BVH_ITERATIVE
					NodeCount = bvhNodeBuffer.Length,
#endif
#if PATH_DEBUGGING
				DebugPaths = (DebugPath*) debugPaths.GetUnsafePtr(),
				DebugCoordinates = int2 (bufferSize / 2)
#endif
				};
			}

			NativeArray<long> timingBuffer = longBuffers.Take();

			JobHandle accumulateJobHandle;
			if (interlacing > 1)
			{
				using (var handles = new NativeArray<JobHandle>(4, Allocator.Temp)
				{
					[0] = new CopyFloat4BufferJob { CancellationToken = cancellationBuffer, Input = colorAccumulationBuffer, Output = colorOutputBuffer }.Schedule(dependency ?? default),
					[1] = new CopyFloat3BufferJob { CancellationToken = cancellationBuffer, Input = normalAccumulationBuffer, Output = normalOutputBuffer }.Schedule(dependency ?? default),
					[2] = new CopyFloat3BufferJob { CancellationToken = cancellationBuffer, Input = albedoAccumulationBuffer, Output = albedoOutputBuffer }.Schedule(dependency ?? default),
					[3] = new ClearBufferJob<Diagnostics> { CancellationToken = cancellationBuffer, Buffer = diagnosticsBuffer }.Schedule(dependency ?? default)
				})
				{
					JobHandle combinedDependencies = JobHandle.CombineDependencies(handles);
					JobHandle startTimerJobHandle = new RecordTimeJob { Buffer = timingBuffer, Index = 0 }.Schedule(combinedDependencies);
					accumulateJobHandle = accumulateJob.Schedule(totalBufferSize, 1, startTimerJobHandle);
				}
			}
			else
			{
				JobHandle startTimerJobHandle = new RecordTimeJob { Buffer = timingBuffer, Index = 0 }.Schedule(dependency ?? default);
				accumulateJobHandle = accumulateJob.Schedule(totalBufferSize, 1, startTimerJobHandle);
			}
			accumulateJobHandle = new RecordTimeJob { Buffer = timingBuffer, Index = 1 }.Schedule(accumulateJobHandle);

			NativeArray<int> reducedRayCountBuffer = intBuffers.Take();
			JobHandle reduceRayCountJobHandle = new ReduceRayCountJob
				{ Diagnostics = diagnosticsBuffer, TotalRayCount = reducedRayCountBuffer }.Schedule(accumulateJobHandle);

			var outputData = new AccumulateOutputData
			{
				Color = float4Buffers.Take(),
				Normal = float3Buffers.Take(),
				Albedo = float3Buffers.Take(),
				ReduceRayCountJobHandle = reduceRayCountJobHandle,
				ReducedRayCount = reducedRayCountBuffer,
				Timing = timingBuffer
			};

			JobHandle combinedDependency = JobHandle.CombineDependencies(
				new CopyFloat4BufferJob { CancellationToken = cancellationBuffer, Input = colorOutputBuffer, Output = outputData.Color }.Schedule(accumulateJobHandle),
				new CopyFloat3BufferJob { CancellationToken = cancellationBuffer, Input = normalOutputBuffer, Output = outputData.Normal }.Schedule(accumulateJobHandle),
				new CopyFloat3BufferJob { CancellationToken = cancellationBuffer, Input = albedoOutputBuffer, Output = outputData.Albedo }.Schedule(accumulateJobHandle));

			NativeArray<float4> colorInputBuffer = colorAccumulationBuffer;
			NativeArray<float3> normalInputBuffer = normalAccumulationBuffer,
				albedoInputBuffer = albedoAccumulationBuffer;

			scheduledAccumulateJobs.Enqueue(new ScheduledJobData<AccumulateOutputData>
			{
				CancellationToken = cancellationBuffer,
				Handle = combinedDependency,
				OutputData = outputData,
				OnComplete = () =>
				{
					float4Buffers.Return(colorInputBuffer);
					float3Buffers.Return(normalInputBuffer);
					float3Buffers.Return(albedoInputBuffer);
					boolBuffers.Return(cancellationBuffer);
				}
			});

			// cycle accumulation output into the next accumulation pass's input
			colorAccumulationBuffer = colorOutputBuffer;
			normalAccumulationBuffer = normalOutputBuffer;
			albedoAccumulationBuffer = albedoOutputBuffer;

			blueNoise.CycleTexture();

			if (AccumulatedSamples + accumulateJob.SampleCount / (float) interlacing >= samplesPerPixel || previewAfterBatch)
				ScheduleCombine(combinedDependency, outputData);

			// schedule another accumulate (but no more than one)
			if (!dependency.HasValue && AccumulatedSamples + accumulateJob.SampleCount / (float) interlacing < samplesPerPixel)
				ScheduleAccumulate(false, JobHandle.CombineDependencies(combinedDependency, reduceRayCountJobHandle));

			JobHandle.ScheduleBatchedJobs();

			if (firstBatch) traceTimer.Restart();
		}

		void ScheduleCombine(JobHandle dependency, AccumulateOutputData accumulateOutput)
		{
			NativeArray<bool> cancellationBuffer = boolBuffers.Take();

			var combineJob = new CombineJob
			{
				CancellationToken = cancellationBuffer,

				DebugMode = denoiseMode == DenoiseMode.None,
				LdrMode = denoiseMode == DenoiseMode.NvidiaOptix,

				InputColor = accumulateOutput.Color,
				InputNormal = accumulateOutput.Normal,
				InputAlbedo = accumulateOutput.Albedo,
				Size = (int2) bufferSize,

				OutputColor = float3Buffers.Take(),
				OutputNormal = float3Buffers.Take(),
				OutputAlbedo = float3Buffers.Take(),
			};

			var totalBufferSize = (int) (bufferSize.x * bufferSize.y);
			JobHandle combineJobHandle = combineJob.Schedule(totalBufferSize, 128, dependency);

			var copyOutputData = new PassOutputData
			{
				Color = float3Buffers.Take(),
				// reuse what we can!
				Normal = accumulateOutput.Normal,
				Albedo = accumulateOutput.Albedo
			};

			JobHandle combinedDependency = JobHandle.CombineDependencies(
				new CopyFloat3BufferJob { CancellationToken = cancellationBuffer, Input = combineJob.OutputColor, Output = copyOutputData.Color }.Schedule(combineJobHandle),
				new CopyFloat3BufferJob { CancellationToken = cancellationBuffer, Input = combineJob.OutputNormal, Output = copyOutputData.Normal }.Schedule(combineJobHandle),
				new CopyFloat3BufferJob { CancellationToken = cancellationBuffer, Input = combineJob.OutputAlbedo, Output = copyOutputData.Albedo }.Schedule(combineJobHandle));

			scheduledCombineJobs.Enqueue(new ScheduledJobData<PassOutputData>
			{
				CancellationToken = cancellationBuffer,
				Handle = combinedDependency,
				OutputData = copyOutputData,
				OnComplete = () =>
				{
					float4Buffers.Return(accumulateOutput.Color);
					float3Buffers.Return(combineJob.OutputColor);
					float3Buffers.Return(combineJob.OutputNormal);
					float3Buffers.Return(combineJob.OutputAlbedo);
					boolBuffers.Return(cancellationBuffer);
				}
			});

			if (denoiseMode != DenoiseMode.None)
				ScheduleDenoise(combinedDependency, copyOutputData);
			else
				ScheduleFinalize(combinedDependency, copyOutputData);
		}

		void ScheduleDenoise(JobHandle dependency, PassOutputData combineOutput)
		{
			NativeArray<float3> denoiseColorOutputBuffer = float3Buffers.Take();
			NativeArray<bool> cancellationBuffer = boolBuffers.Take();

			JobHandle denoiseJobHandle = default;

			JobHandle combinedDependency = dependency;
			foreach (ScheduledJobData<PassOutputData> priorFinalizeJob in scheduledDenoiseJobs)
				combinedDependency = JobHandle.CombineDependencies(combinedDependency, priorFinalizeJob.Handle);

			switch (denoiseMode)
			{
				case DenoiseMode.OpenImageDenoise:
				{
					var denoiseJob = new OpenImageDenoiseJob
					{
						CancellationToken = cancellationBuffer,
						Width = (ulong) bufferSize.x,
						Height = (ulong) bufferSize.y,
						InputColor = combineOutput.Color,
						InputNormal = combineOutput.Normal,
						InputAlbedo = combineOutput.Albedo,
						OutputColor = denoiseColorOutputBuffer,
						DenoiseFilter = oidnFilter
					};
					denoiseJobHandle = denoiseJob.Schedule(combinedDependency);
					break;
				}

				case DenoiseMode.NvidiaOptix:
				{
					var denoiseJob = new OptixDenoiseJob
					{
						CancellationToken = cancellationBuffer,
						Denoiser = optixDenoiser,
						CudaStream = cudaStream,
						InputColor = combineOutput.Color,
						InputAlbedo = combineOutput.Albedo,
						OutputColor = denoiseColorOutputBuffer,
						BufferSize = (uint2) bufferSize,
						DenoiserState = optixDenoiserState,
						ScratchMemory = optixScratchMemory,
						DenoiserSizes = optixDenoiserSizes,
						InputAlbedoBuffer = optixAlbedoBuffer,
						InputColorBuffer = optixColorBuffer,
						OutputColorBuffer = optixOutputBuffer
					};
					denoiseJobHandle = denoiseJob.Schedule(combinedDependency);
					break;
				}
			}

			var copyOutputData = new PassOutputData
			{
				Color = combineOutput.Color,
				Normal = combineOutput.Normal,
				Albedo = combineOutput.Albedo
			};

			JobHandle copyJobHandle = new CopyFloat3BufferJob
				{ CancellationToken = cancellationBuffer, Input = denoiseColorOutputBuffer, Output = copyOutputData.Color }.Schedule(denoiseJobHandle);

			scheduledDenoiseJobs.Enqueue(new ScheduledJobData<PassOutputData>
			{
				CancellationToken = cancellationBuffer,
				Handle = copyJobHandle,
				OutputData = copyOutputData,
				OnComplete = () =>
				{
					float3Buffers.Return(denoiseColorOutputBuffer);
					boolBuffers.Return(cancellationBuffer);
				}
			});

			ScheduleFinalize(copyJobHandle, copyOutputData);
		}

		void ScheduleFinalize(JobHandle dependency, PassOutputData lastPassOutput)
		{
			NativeArray<bool> cancellationBuffer = boolBuffers.Take();

			var finalizeJob = new FinalizeTexturesJob
			{
				CancellationToken = cancellationBuffer,

				InputColor = lastPassOutput.Color,
				InputNormal = lastPassOutput.Normal,
				InputAlbedo = lastPassOutput.Albedo,

				OutputColor = frontBuffer,
				OutputNormal = normalsBuffer,
				OutputAlbedo = albedoBuffer
			};

			var totalBufferSize = (int) (bufferSize.x * bufferSize.y);

			JobHandle combinedDependency = dependency;
			foreach (ScheduledJobData<PassOutputData> priorFinalizeJob in scheduledFinalizeJobs)
				combinedDependency = JobHandle.CombineDependencies(combinedDependency, priorFinalizeJob.Handle);

			JobHandle finalizeJobHandle = finalizeJob.Schedule(totalBufferSize, 128, combinedDependency);

			scheduledFinalizeJobs.Enqueue(new ScheduledJobData<PassOutputData>
			{
				CancellationToken = cancellationBuffer,
				Handle = finalizeJobHandle,
				OnComplete = () =>
				{
					float3Buffers.Return(lastPassOutput.Color);
					float3Buffers.Return(lastPassOutput.Normal);
					float3Buffers.Return(lastPassOutput.Albedo);
					boolBuffers.Return(cancellationBuffer);
				}
			});
		}

		void CleanCamera()
		{
#if UNITY_EDITOR
			targetCamera.transform.hasChanged = false;
			scene.UpdateFromGameView();
#endif
		}

		unsafe void SwapBuffers()
		{
			float bufferMin = float.MaxValue, bufferMax = float.MinValue;
			Diagnostics* diagnosticsPtr = (Diagnostics*) NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(diagnosticsBuffer);

			switch (bufferView)
			{
				case BufferView.RayCount:
					for (int i = 0; i < diagnosticsBuffer.Length; i++, ++diagnosticsPtr)
					{
						Diagnostics value = *diagnosticsPtr;
						bufferMin = min(bufferMin, value.RayCount);
						bufferMax = max(bufferMax, value.RayCount);
					}

					break;

#if FULL_DIAGNOSTICS && BVH_ITERATIVE
				case BufferView.BvhHitCount:
					for (int i = 0; i < diagnosticsBuffer.Length; i++, ++diagnosticsPtr)
					{
						Diagnostics value = *diagnosticsPtr;
						bufferMin = min(bufferMin, value.BoundsHitCount);
						bufferMax = max(bufferMax, value.BoundsHitCount);
					}
					break;

				case BufferView.CandidateCount:
					for (int i = 0; i < diagnosticsBuffer.Length; i++, ++diagnosticsPtr)
					{
						Diagnostics value = *diagnosticsPtr;
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

				RebuildOptixBuffers((uint2) lastBufferSize);
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

		unsafe void RebuildOptixBuffers(uint2 lastBufferSize)
		{
			var newBufferSize = (uint2) bufferSize;

			SizeT lastSizeInBytes = lastBufferSize.x * lastBufferSize.y * sizeof(float3);
			SizeT newSizeInBytes = newBufferSize.x * newBufferSize.y * sizeof(float3);

			void Check(CudaError cudaError)
			{
				if (cudaError != CudaError.Success)
					Debug.LogError($"CUDA Error : {cudaError}");
			}

			Check(optixColorBuffer.EnsureCapacity(lastSizeInBytes, newSizeInBytes));
			Check(optixAlbedoBuffer.EnsureCapacity(lastSizeInBytes, newSizeInBytes));
			Check(optixOutputBuffer.EnsureCapacity(lastSizeInBytes, newSizeInBytes));

			OptixDenoiserSizes lastSizes = optixDenoiserSizes;
			OptixDenoiserSizes newSizes = default;
			OptixDenoiser.ComputeMemoryResources(optixDenoiser, newBufferSize.x, newBufferSize.y, &newSizes);

			Check(optixScratchMemory.EnsureCapacity(lastSizes.RecommendedScratchSizeInBytes,
				newSizes.RecommendedScratchSizeInBytes));
			Check(optixDenoiserState.EnsureCapacity(lastSizes.StateSizeInBytes, newSizes.StateSizeInBytes));

			optixDenoiserSizes = newSizes;

			OptixDenoiser.Setup(optixDenoiser, cudaStream, newBufferSize.x, newBufferSize.y, optixDenoiserState,
				newSizes.StateSizeInBytes, optixScratchMemory, newSizes.RecommendedScratchSizeInBytes);
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

			perlinNoise.Generate(scene.RandomSeed);

			worldNeedsRebuild = false;
		}

		unsafe void RebuildEntityBuffers()
		{
			int entityCount = ActiveEntities.Count;

			entityBuffer.EnsureCapacity(entityCount);

			sphereBuffer.EnsureCapacity(ActiveEntities.Count(x => x.Type == EntityType.Sphere));
			rectBuffer.EnsureCapacity(ActiveEntities.Count(x => x.Type == EntityType.Rect));
			boxBuffer.EnsureCapacity(ActiveEntities.Count(x => x.Type == EntityType.Box));

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
				TextureData albedo = materialData.Albedo;
				TextureData emission = materialData.Emission;

				Material material = materialData
					? new Material(materialData.Type, materialData.TextureScale * sizeFactor,
						albedo.GetRuntimeData(), emission.GetRuntimeData(),
						materialData.Roughness, materialData.RefractiveIndex, materialData.Density)
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
		unsafe void RebuildBvh()
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
		unsafe bool HitWorld(Ray r, out HitRecord hitRec)
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
						(float lambertian, float metal, float dielectric, float light) probabilities = (
							group.LambertChance,
							group.MetalChance,
							group.DieletricChance,
							group.LightChance);

						float sum = probabilities.lambertian + probabilities.metal + probabilities.dielectric + probabilities.light;
						probabilities.metal += probabilities.lambertian;
						probabilities.dielectric += probabilities.metal;
						probabilities.light += probabilities.dielectric;
						probabilities.lambertian /= sum;
						probabilities.metal /= sum;
						probabilities.dielectric /= sum;
						probabilities.light /= sum;

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
						else if (randomValue < probabilities.light)
						{
							Color from = group.Emissive.colorKeys[0].color;
							Color to = group.Emissive.colorKeys[1].color;
							float3 color = rng.NextFloat3(from.ToFloat3(), to.ToFloat3());
							material = MaterialData.DiffuseLight(TextureData.Constant(color));
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

								if (group.OffsetByRadius)
									center += radius;

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