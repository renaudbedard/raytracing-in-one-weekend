using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
using ReadOnly = Sirenix.OdinInspector.ReadOnlyAttribute;
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
        [SerializeField] [Range(1, 100)] int traceDepth = 20;
        [SerializeField] bool previewAfterBatch = true;
        [SerializeField] bool stopWhenCompleted = true;

        [Title("Camera")] 
        [SerializeField] [Range(0, 180)] float fieldOfView = 90;
        [SerializeField] float3 cameraOrigin = new float3(3, 3, 2);
        [SerializeField] float3 cameraLookAt = new float3(0, 0, -1);
        [SerializeField] float cameraAperture = 2;

        [Title("World")]
        [SerializeField] SphereData[] spheres = null;

        [Title("Debug")]
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
        [ShowInInspector] [InlineEditor(InlineEditorModes.LargePreview)] [ReadOnly]  
#else
        public
#endif
        Texture2D frontBuffer;

#if ODIN_INSPECTOR && UNITY_EDITOR
        [Button(Name = "Save")]
        [DisableInEditorMode]
        void Save()
        {
            byte[] pngBytes = frontBuffer.EncodeToPNG();
            File.WriteAllBytes(
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                    $"Raytracer {DateTime.Now:yyyy-MM-dd HH-mm-ss}.png"), pngBytes);
        }
#endif
        
        CommandBuffer commandBuffer;
        NativeArray<float4> accumulationInputBuffer, accumulationOutputBuffer;
        NativeArray<half4> backBuffer;
        NativeArray<Primitive> primitiveBuffer;
        NativeArray<Sphere> sphereBuffer;

        JobHandle? accumulateJobHandle;
        JobHandle? combineJobHandle;
        bool commandBufferHooked;
        bool worldNeedsRebuild;        

        readonly Stopwatch traceTimer = new Stopwatch();
        readonly List<SphereData> activeSpheres = new List<SphereData>();

        int bufferWidth, bufferHeight;

        void Awake()
        {
            commandBuffer = new CommandBuffer { name = "Raytracer" };

            frontBuffer = new Texture2D(0, 0, TextureFormat.RGBAHalf, false)
            {
                hideFlags = HideFlags.HideAndDontSave
            };

            RebuildWorld();
            EnsureBuffersBuilt();
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

            if (backBuffer.IsCreated) backBuffer.Dispose();
            if (primitiveBuffer.IsCreated) primitiveBuffer.Dispose();
            if (sphereBuffer.IsCreated) sphereBuffer.Dispose();
            if (accumulationInputBuffer.IsCreated) accumulationInputBuffer.Dispose();
            if (accumulationOutputBuffer.IsCreated) accumulationOutputBuffer.Dispose();
        }

        void Update()
        {
#if UNITY_EDITOR
            // watch for material data changes (won't catch those from OnValidate)
            if (spheres.Any(x => x.Material.Dirty))
            {
                foreach (var sphere in spheres) sphere.Material.Dirty = false;
                worldNeedsRebuild = true;
            }
#endif
            
            int currentWidth = Mathf.RoundToInt(targetCamera.pixelWidth * resolutionScaling);
            int currentHeight = Mathf.RoundToInt(targetCamera.pixelHeight * resolutionScaling);
            
            bool noActiveJob = accumulateJobHandle == null && combineJobHandle == null;
            bool jobCompleted = (accumulateJobHandle?.IsCompleted ?? false) ||
                                (combineJobHandle?.IsCompleted ?? false);
            bool buffersNeedRebuild = currentWidth != bufferWidth || currentHeight != bufferHeight;
            
            if ((noActiveJob || jobCompleted) && (worldNeedsRebuild || buffersNeedRebuild))
            {
                accumulateJobHandle?.Complete();
                combineJobHandle?.Complete();
                accumulateJobHandle = combineJobHandle = null;

                if (worldNeedsRebuild)
                    RebuildWorld();
                
                EnsureBuffersBuilt();

                ScheduleAccumulate(true);
            }

            if (accumulateJobHandle.HasValue && accumulateJobHandle.Value.IsCompleted)
            {
                accumulateJobHandle.Value.Complete();
                accumulateJobHandle = null;
                
                accumulatedSamples += samplesPerBatch;
                ForceUpdateInspector();

                if (accumulatedSamples >= samplesPerPixel || previewAfterBatch)
                    ScheduleCombine();
                else
                    ScheduleAccumulate(false);
            }

            if (combineJobHandle.HasValue && combineJobHandle.Value.IsCompleted)
            {
                combineJobHandle.Value.Complete();
                combineJobHandle = null;

                bool traceCompleted = false;
                if (accumulatedSamples >= samplesPerPixel)
                {
                    traceCompleted = true;
                    lastTraceDuration = (float) traceTimer.ElapsedTicks / TimeSpan.TicksPerMillisecond;
                }

                SwapBuffers();
                ForceUpdateInspector();

                if (!(traceCompleted && stopWhenCompleted))
                    ScheduleAccumulate(traceCompleted);
            }
        }

        void ForceUpdateInspector()
        {
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(this);
#endif
        }

        void SwapBuffers()
        {
            frontBuffer.LoadRawTextureData(backBuffer);
            frontBuffer.Apply(false);

            if (!commandBufferHooked)
            {
                targetCamera.AddCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
                commandBufferHooked = true;
            }
        }

        void ScheduleAccumulate(bool firstBatch)
        {
            float focalDistance = length(cameraOrigin - cameraLookAt);

            var raytracingCamera = new Camera(cameraOrigin, cameraLookAt, float3(0, 1, 0),
                fieldOfView, (float) bufferWidth / bufferHeight, cameraAperture, focalDistance);

            if (firstBatch)
            {
                accumulationInputBuffer.Dispose();
                accumulationInputBuffer = new NativeArray<float4>(bufferWidth * bufferHeight, Allocator.Persistent);

                accumulatedSamples = 0;
                ForceUpdateInspector();

                traceTimer.Restart();
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
                Primitives = primitiveBuffer
            };
            accumulateJobHandle = job.Schedule(bufferWidth * bufferHeight, bufferWidth);

            JobHandle.ScheduleBatchedJobs();
        }

        void ScheduleCombine()
        {
            var job = new CombineJob
            {
                Input = accumulationOutputBuffer,
                Output = backBuffer
            };
            combineJobHandle = job.Schedule(bufferWidth * bufferHeight, bufferWidth);

            JobHandle.ScheduleBatchedJobs();
        }

        void EnsureBuffersBuilt()
        {
            int width = Mathf.RoundToInt(targetCamera.pixelWidth * resolutionScaling);
            int height = Mathf.RoundToInt(targetCamera.pixelHeight * resolutionScaling);
            
            if (frontBuffer.width != width || frontBuffer.height != height)
            {
                if (commandBufferHooked)
                {
                    targetCamera.RemoveCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
                    commandBufferHooked = false;
                }

                frontBuffer.Resize(width, height);
                frontBuffer.filterMode = resolutionScaling > 1 ? FilterMode.Bilinear : FilterMode.Point;

                commandBuffer.Clear();
                commandBuffer.Blit(frontBuffer, new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget));
            }

            if (backBuffer.Length != width * height)
            {
                if (backBuffer.IsCreated) backBuffer.Dispose();
                backBuffer = new NativeArray<half4>(width * height,
                    Allocator.Persistent,
                    NativeArrayOptions.UninitializedMemory);
            }
            
            if (accumulationInputBuffer.Length != width * height)
            {
                if (accumulationInputBuffer.IsCreated) accumulationInputBuffer.Dispose();
                accumulationInputBuffer = new NativeArray<float4>(width * height, Allocator.Persistent);
            }
            
            if (accumulationOutputBuffer.Length != width * height)
            {
                if (accumulationOutputBuffer.IsCreated) accumulationOutputBuffer.Dispose();
                accumulationOutputBuffer = new NativeArray<float4>(width * height,
                    Allocator.Persistent,
                    NativeArrayOptions.UninitializedMemory);
            }

            bufferWidth = width;
            bufferHeight = height;
        }

        void RebuildWorld()
        {
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
                if (primitiveBuffer.IsCreated)
                    primitiveBuffer.Dispose();

                primitiveBuffer = new NativeArray<Primitive>(primitiveCount, Allocator.Persistent);
            }

            // rebuild individual typed primitive buffers
            if (!sphereBuffer.IsCreated || sphereBuffer.Length != activeSpheres.Count)
            {
                if (sphereBuffer.IsCreated)
                    sphereBuffer.Dispose();

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
        }

        static void ExchangeBuffers(ref NativeArray<float4> lhs, ref NativeArray<float4> rhs)
        {
            var temp = lhs;
            lhs = rhs;
            rhs = temp;
        }
    }
}