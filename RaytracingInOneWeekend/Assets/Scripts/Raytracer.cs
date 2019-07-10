using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;
using Debug = UnityEngine.Debug;
using Random = Unity.Mathematics.Random;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
using ReadOnly = Sirenix.OdinInspector.ReadOnlyAttribute;
using System.IO;
#else
using Title = UnityEngine.HeaderAttribute;
#endif

namespace RaytracerInOneWeekend
{
    public class Raytracer : MonoBehaviour
    {
        [Title("References")]
        [SerializeField] UnityEngine.Camera targetCamera = null;

        [Title("Settings")]
        [SerializeField] [Range(0.01f, 2)] float resolutionScaling = 0.5f;
        [SerializeField] [Range(1, 2000)] int samplesPerPixel = 2000;
        [SerializeField] [Range(1, 100)] int samplesPerBatch = 10;
        [SerializeField] [Range(1, 100)] int traceDepth = 35;
        [SerializeField] bool previewAfterBatch = true;
        [SerializeField] bool stopWhenCompleted = true;

        [Title("Camera")]
        [SerializeField] float cameraAperture = 0.1f;

        [Title("World")]
        [SerializeField] bool randomScene = true;
#if ODIN_INSPECTOR
        [ShowIf(nameof(randomScene))]
#endif
        [SerializeField] uint sceneSeed = 45573880;
#if ODIN_INSPECTOR
        [HideIf(nameof(randomScene))]
#endif
        [SerializeField] SphereData[] spheres = null;

        [Title("Debug")]
#if ODIN_INSPECTOR
        [ShowInInspector] [ReadOnly]
#else
        public
#endif
        float millionRaysPerSecond;
#if ODIN_INSPECTOR
        [ShowInInspector] [ReadOnly]
#else
        public
#endif
        float lastBatchDuration;
#if ODIN_INSPECTOR
        [ShowInInspector] [ReadOnly]
#else
        public
#endif
        float lastTraceDuration;
#if ODIN_INSPECTOR
        [ShowInInspector] [ReadOnly]
#else
        public
#endif
        int accumulatedSamples;
#if ODIN_INSPECTOR
        [DisableIf(nameof(TraceActive))] [DisableInEditorMode] [Button]
        void TriggerTrace() => ScheduleAccumulate(true);
#endif
#if ODIN_INSPECTOR
        [ShowInInspector] [InlineEditor(InlineEditorModes.LargePreview)] [ReadOnly]
#else
        public
#endif
        Texture2D frontBufferTexture;

        CommandBuffer commandBuffer;
        NativeArray<float4> accumulationInputBuffer, accumulationOutputBuffer;
        NativeArray<half4> frontBuffer;
        NativeArray<Primitive> primitiveBuffer;
        NativeArray<int> rayCountBuffer;
        NativeArray<Sphere> sphereBuffer;

        JobHandle? accumulateJobHandle;
        JobHandle? combineJobHandle;
        bool commandBufferHooked;
        bool worldNeedsRebuild;
        float lastFieldOfView, lastFocusDistance;
        bool initialized;

        readonly Stopwatch batchTimer = new Stopwatch();
        readonly Stopwatch traceTimer = new Stopwatch();

        readonly List<SphereData> activeSpheres = new List<SphereData>();

        int bufferWidth, bufferHeight;

        internal NativeArray<Primitive> Primitives => primitiveBuffer;
        internal bool TraceActive => accumulateJobHandle.HasValue || combineJobHandle.HasValue;

        void Awake()
        {
            commandBuffer = new CommandBuffer { name = "Raytracer" };
            frontBufferTexture = new Texture2D(0, 0, TextureFormat.RGBAHalf, false)
            {
                hideFlags = HideFlags.HideAndDontSave
            };
        }

        void Start()
        {
            RebuildWorld();
            EnsureBuffersBuilt();
            CleanCamera();

            ScheduleAccumulate(true);
        }

#if UNITY_EDITOR
        void OnValidate()
        {
            if (Application.isPlaying)
                worldNeedsRebuild = true;
        }
#endif

        void OnDestroy()
        {
            // if there is a running job, let it know it needs to cancel and wait for completion
            accumulateJobHandle?.Complete();
            combineJobHandle?.Complete();

            if (primitiveBuffer.IsCreated) primitiveBuffer.Dispose();
            if (sphereBuffer.IsCreated) sphereBuffer.Dispose();
            if (accumulationInputBuffer.IsCreated) accumulationInputBuffer.Dispose();
            if (accumulationOutputBuffer.IsCreated) accumulationOutputBuffer.Dispose();
            if (rayCountBuffer.IsCreated) rayCountBuffer.Dispose();
        }

        void Update()
        {
#if UNITY_EDITOR
            // watch for material data changes (won't catch those from OnValidate)
            if (!randomScene && spheres.Any(x => x.Material.Dirty))
            {
                foreach (var sphere in spheres) sphere.Material.Dirty = false;
                worldNeedsRebuild = true;
            }
#endif

            int currentWidth = Mathf.RoundToInt(targetCamera.pixelWidth * resolutionScaling);
            int currentHeight = Mathf.RoundToInt(targetCamera.pixelHeight * resolutionScaling);

            bool buffersNeedRebuild = currentWidth != bufferWidth || currentHeight != bufferHeight;
            bool cameraDirty = targetCamera.transform.hasChanged ||
                               !Mathf.Approximately(lastFieldOfView, targetCamera.fieldOfView);
            bool traceNeedsReset = buffersNeedRebuild || worldNeedsRebuild || cameraDirty;

            void RebuildDirtyComponents()
            {
                if (buffersNeedRebuild) EnsureBuffersBuilt();
                if (worldNeedsRebuild) RebuildWorld();
                if (cameraDirty) CleanCamera();
            }

            if (!TraceActive && traceNeedsReset)
            {
                RebuildDirtyComponents();
                ScheduleAccumulate(true);
            }

            if (accumulateJobHandle.HasValue && accumulateJobHandle.Value.IsCompleted)
            {
                accumulateJobHandle.Value.Complete();
                accumulateJobHandle = null;

                var elapsedTime = batchTimer.Elapsed;
                int rayCount = rayCountBuffer.Sum();

                accumulatedSamples += samplesPerBatch;
                lastBatchDuration = (float) elapsedTime.TotalMilliseconds;
                millionRaysPerSecond = rayCount / (float) elapsedTime.TotalSeconds / 1000000;
                ForceUpdateInspector();

                if (accumulatedSamples >= samplesPerPixel || previewAfterBatch)
                    ScheduleCombine();
                else
                {
                    if (!previewAfterBatch)
                        RebuildDirtyComponents();

                    ScheduleAccumulate(traceNeedsReset && !previewAfterBatch);
                }
            }

            if (combineJobHandle.HasValue && combineJobHandle.Value.IsCompleted)
            {
                combineJobHandle.Value.Complete();
                combineJobHandle = null;

                bool traceCompleted = false;
                if (accumulatedSamples >= samplesPerPixel)
                {
                    traceCompleted = true;
                    lastTraceDuration = (float) traceTimer.Elapsed.TotalMilliseconds;
                }

                SwapBuffers();
                ForceUpdateInspector();

                RebuildDirtyComponents();

                if (!(traceCompleted && stopWhenCompleted) || traceNeedsReset)
                    ScheduleAccumulate(traceCompleted | traceNeedsReset);
            }
        }

        void ForceUpdateInspector()
        {
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(this);
#endif
        }

#if ODIN_INSPECTOR && UNITY_EDITOR
        [Button] [DisableInEditorMode]
        void SaveFrontBuffer()
        {
            byte[] pngBytes = frontBufferTexture.EncodeToPNG();
            File.WriteAllBytes(
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                    $"Raytracer {DateTime.Now:yyyy-MM-dd HH-mm-ss}.png"), pngBytes);
        }
#endif

        void CleanCamera()
        {
            lastFieldOfView = targetCamera.fieldOfView;
            targetCamera.transform.hasChanged = false;
        }

        void SwapBuffers()
        {
            frontBufferTexture.Apply(false);

            if (!commandBufferHooked)
            {
                targetCamera.AddCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
                commandBufferHooked = true;
            }
        }

        void ScheduleAccumulate(bool firstBatch)
        {
            var cameraTransform = targetCamera.transform;
            var origin = cameraTransform.localPosition;
            var lookAt = origin + cameraTransform.forward;
            var focusDistance = lastFocusDistance;

            if (primitiveBuffer.Hit(new Ray(origin, cameraTransform.forward), 0, float.PositiveInfinity,
                out HitRecord hitRec))
            {
                lastFocusDistance = focusDistance = hitRec.Distance;
            }

            var raytracingCamera = new Camera(origin, lookAt, cameraTransform.up, targetCamera.fieldOfView,
                (float) bufferWidth / bufferHeight, cameraAperture, focusDistance);

            if (rayCountBuffer.IsCreated) rayCountBuffer.Dispose();
            rayCountBuffer = new NativeArray<int>(bufferWidth * bufferHeight, Allocator.Persistent);

            if (firstBatch)
            {
                if (accumulationInputBuffer.IsCreated) accumulationInputBuffer.Dispose();
                accumulationInputBuffer = new NativeArray<float4>(bufferWidth * bufferHeight, Allocator.Persistent);

                accumulatedSamples = 0;
                ForceUpdateInspector();
            }
            else
                ExchangeBuffers(ref accumulationInputBuffer, ref accumulationOutputBuffer);

            var job = new AccumulateJob
            {
                Size = int2(bufferWidth, bufferHeight),
                Camera = raytracingCamera,
                InputSamples = accumulationInputBuffer,
                OutputSamples = accumulationOutputBuffer,
                Seed = (uint) Time.frameCount + 1,
                SampleCount = Math.Min(samplesPerPixel, samplesPerBatch),
                TraceDepth = traceDepth,
                Primitives = primitiveBuffer,
                OutputRayCount = rayCountBuffer
            };

            accumulateJobHandle = job.Schedule(bufferWidth * bufferHeight, 1);

            batchTimer.Restart();
            if (firstBatch) traceTimer.Restart();
            JobHandle.ScheduleBatchedJobs();
        }

        void ScheduleCombine()
        {
            var job = new CombineJob
            {
                Input = accumulationOutputBuffer,
                Output = frontBuffer
            };

            combineJobHandle = job.Schedule(bufferWidth * bufferHeight, 128);

            JobHandle.ScheduleBatchedJobs();
        }

        void EnsureBuffersBuilt()
        {
            int width = Mathf.RoundToInt(targetCamera.pixelWidth * resolutionScaling);
            int height = Mathf.RoundToInt(targetCamera.pixelHeight * resolutionScaling);

            if (frontBufferTexture.width != width || frontBufferTexture.height != height)
            {
                if (commandBufferHooked)
                {
                    targetCamera.RemoveCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
                    commandBufferHooked = false;
                }

                frontBufferTexture.Resize(width, height);
                frontBufferTexture.filterMode = resolutionScaling > 1 ? FilterMode.Bilinear : FilterMode.Point;
                frontBuffer = frontBufferTexture.GetRawTextureData<half4>();

                commandBuffer.Clear();
                commandBuffer.Blit(frontBufferTexture, new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget));

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

            bufferWidth = width;
            bufferHeight = height;
        }

        void RebuildWorld()
        {
            if (randomScene)
            {
                BuildRandomScene();
                worldNeedsRebuild = false;
                return;
            }

            int primitiveCount = 0;

            activeSpheres.Clear();
            foreach (SphereData sphere in spheres)
                if (sphere.Enabled)
                    activeSpheres.Add(sphere);

            primitiveCount += activeSpheres.Count;

            // other typed active primitives would be collected here

            // rebuild primitive buffer
            if (!primitiveBuffer.IsCreated || primitiveBuffer.Length != primitiveCount)
            {
                if (primitiveBuffer.IsCreated) primitiveBuffer.Dispose();
                primitiveBuffer = new NativeArray<Primitive>(primitiveCount, Allocator.Persistent);
            }

            // rebuild individual typed primitive buffers
            if (!sphereBuffer.IsCreated || sphereBuffer.Length != activeSpheres.Count)
            {
                if (sphereBuffer.IsCreated) sphereBuffer.Dispose();
                sphereBuffer = new NativeArray<Sphere>(activeSpheres.Count, Allocator.Persistent);
            }

            // collect primitives
            int primitiveIndex = 0;
            for (var i = 0; i < activeSpheres.Count; i++)
            {
                var sphereData = activeSpheres[i];
                var materialData = sphereData.Material;
                sphereBuffer[i] = new Sphere(sphereData.Center, sphereData.Radius,
                    new Material(materialData.Type, materialData.Albedo.ToFloat3(), materialData.Fuzz,
                        materialData.RefractiveIndex));

                var sphereSlice = new NativeSlice<Sphere>(sphereBuffer, i, 1);
                primitiveBuffer[primitiveIndex++] = new Primitive(sphereSlice);
            }

            worldNeedsRebuild = false;

            Debug.Log("Rebuilt world");
        }

        void BuildRandomScene()
        {
            int n = 500;

            if (sphereBuffer.IsCreated) sphereBuffer.Dispose();
            sphereBuffer = new NativeArray<Sphere>(n, Allocator.Persistent)
            {
                [0] = new Sphere(new float3(0, -1000, 0), 1000, Material.Lambertian(0.5f))
            };

            var rng = new Random(sceneSeed);

            int sphereIndex = 1;
            for (int a = -11; a < 11; a++)
            {
                for (int b = -11; b < 11; b++)
                {
                    float materialProb = rng.NextFloat();
                    float3 center = float3(a + 0.9f * rng.NextFloat(), 0.2f, b + 0.9f * rng.NextFloat());

                    if (distance(center, float3(4, 0.2f, 0)) <= 0.9)
                        continue;

                    if (materialProb < 0.8)
                        sphereBuffer[sphereIndex++] = new Sphere(center, 0.2f, Material.Lambertian(rng.NextFloat3() * rng.NextFloat3()));
                    else if (materialProb < 0.95)
                        sphereBuffer[sphereIndex++] = new Sphere(center, 0.2f,
                            Material.Metal(rng.NextFloat3(0.5f, 1), rng.NextFloat(0, 0.5f)));
                    else
                        sphereBuffer[sphereIndex++] = new Sphere(center, 0.2f, Material.Dielectric(1.5f));
                }
            }

            sphereBuffer[sphereIndex++] = new Sphere(float3(0, 1, 0), 1, Material.Dielectric(1.5f));
            sphereBuffer[sphereIndex++] = new Sphere(float3(-4, 1, 0), 1, Material.Lambertian(float3(0.4f, 0.2f, 0.1f)));
            sphereBuffer[sphereIndex++] = new Sphere(float3(4, 1, 0), 1, Material.Metal(float3(0.7f, 0.6f, 0.5f)));

            int sphereCount = sphereIndex;

            if (primitiveBuffer.IsCreated) primitiveBuffer.Dispose();
            primitiveBuffer = new NativeArray<Primitive>(sphereCount, Allocator.Persistent);

            int primitiveIndex = 0;
            for (var i = 0; i < sphereCount; i++)
            {
                var sphereSlice = new NativeSlice<Sphere>(sphereBuffer, i, 1);
                primitiveBuffer[primitiveIndex++] = new Primitive(sphereSlice);
            }

            Debug.Log("Rebuilt random scene");
        }

        static void ExchangeBuffers(ref NativeArray<float4> lhs, ref NativeArray<float4> rhs)
        {
            var temp = lhs;
            lhs = rhs;
            rhs = temp;
        }
    }
}