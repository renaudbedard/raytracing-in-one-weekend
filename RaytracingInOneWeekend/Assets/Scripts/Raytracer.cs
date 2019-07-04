using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Sirenix.OdinInspector;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;
using ReadOnly = Sirenix.OdinInspector.ReadOnlyAttribute;

namespace RaytracerInOneWeekend
{
    public class Raytracer : MonoBehaviour
    {
        [Title("References")]
        [SerializeField] UnityEngine.Camera targetCamera = null;

        [Title("Settings")]
        [SerializeField] [Range(0.01f, 2)] float resolutionScaling = 1;
        [SerializeField] [Range(1, 1000)] int samplesPerPixel = 100;
        [SerializeField] [Range(1, 100)] int traceDepth = 50;

        [Title("World")] [SerializeField] SphereData[] spheres = null;
        
        [Title("Debug")]
        [ShowInInspector] [ReadOnly] float lastJobDuration;
        [ShowInInspector] [InlineEditor(InlineEditorModes.LargePreview)] [ReadOnly] Texture2D frontBuffer;

        CommandBuffer commandBuffer;
        NativeArray<half4> backBuffer;
        NativeArray<Sphere> sphereBuffer;

        JobHandle? raytraceJobHandle;
        bool commandBufferHooked;

        readonly Stopwatch jobTimer = new Stopwatch();

        int Width => Mathf.RoundToInt(targetCamera.pixelWidth * resolutionScaling);
        int Height => Mathf.RoundToInt(targetCamera.pixelHeight * resolutionScaling);

        void Awake()
        {
            commandBuffer = new CommandBuffer { name = "Raytracer" };

            frontBuffer = new Texture2D(0, 0, TextureFormat.RGBAHalf, false)
            {
                hideFlags = HideFlags.HideAndDontSave
            };

            EnsureBuffersBuilt();
            RebuildWorld();
            ScheduleJob();
        }
        
#if UNITY_EDITOR
        bool worldNeedsRebuild;
        void OnValidate()
        {
            if (Application.isPlaying)
            {
                // we COULD do more fine-grained dirty-checking here
                worldNeedsRebuild = true;
            }
        }
#endif

        void OnDestroy()
        {
            // if there is a running job, wait for completion so that we can dispose buffers
            raytraceJobHandle?.Complete();
            
            if (backBuffer.IsCreated) backBuffer.Dispose();
            if (sphereBuffer.IsCreated) sphereBuffer.Dispose();
        }

        void Update()
        {
            if (raytraceJobHandle.HasValue)
            {
                // don't actively wait for it, just poll completion status
                if (raytraceJobHandle.Value.IsCompleted)
                {
                    // though we do need to call Complete to regain ownership of buffers
                    raytraceJobHandle.Value.Complete();
                    lastJobDuration = (float) jobTimer.ElapsedTicks / TimeSpan.TicksPerMillisecond;
                    SwapBuffers();
                }
                else
                {
                    // we already have a job in progress; early out
                    return;
                }
            }
            
#if UNITY_EDITOR
            // watch for material data changes (won't catch those from OnValidate)
            if (spheres.Any(x => x.Material.Dirty))
            {
                foreach (var sphere in spheres) sphere.Material.Dirty = false;
                worldNeedsRebuild = true;
            }
            
            // watch for local field changes
            if (worldNeedsRebuild)
            {
                RebuildWorld();
                worldNeedsRebuild = false;
            }
#endif
            EnsureBuffersBuilt();
            ScheduleJob();
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

        void ScheduleJob()
        {
            float aspect = (float) Width / Height;

            var raytracingCamera = new Camera(0,
                float3(-aspect, -1, -1),
                float3(aspect * 2, 0, 0),
                float3(0, 2, 0));

            var raytraceJob = new RaytraceJob
            {
                Size = int2(Width, Height),
                Camera = raytracingCamera,
                Target = backBuffer,
                Spheres = sphereBuffer,
                Rng = new Random((uint) Time.frameCount + 1),
                SampleCount = samplesPerPixel,
                TraceDepth = traceDepth
            };
            raytraceJobHandle = raytraceJob.Schedule(Width * Height, Width);
            
            // kick the job system
            JobHandle.ScheduleBatchedJobs();
            
            jobTimer.Restart();
        }
        
        void EnsureBuffersBuilt()
        {
            if (frontBuffer.width != Width || frontBuffer.height != Height)
            {
                if (commandBufferHooked)
                {
                    targetCamera.RemoveCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
                    commandBufferHooked = false;
                }

                frontBuffer.Resize(Width, Height);
                frontBuffer.filterMode = resolutionScaling > 1 ? FilterMode.Bilinear : FilterMode.Point;

                commandBuffer.Clear();
                commandBuffer.Blit(frontBuffer, new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget));
            }

            if (backBuffer.Length != Width * Height)
            {
                if (backBuffer.IsCreated)
                    backBuffer.Dispose();

                backBuffer = new NativeArray<half4>(Width * Height,
                    Allocator.Persistent,
                    NativeArrayOptions.UninitializedMemory);
            }
        }

        readonly List<SphereData> activeSpheres = new List<SphereData>();
        void RebuildWorld()
        {
            activeSpheres.Clear();
            foreach (var sphere in spheres)
                if (sphere.Enabled)
                    activeSpheres.Add(sphere);

            if (!sphereBuffer.IsCreated || sphereBuffer.Length != activeSpheres.Count)
            {
                if (sphereBuffer.IsCreated)
                    sphereBuffer.Dispose();
                
                sphereBuffer = new NativeArray<Sphere>(activeSpheres.Count, Allocator.Persistent);
            }

            for (var i = 0; i < activeSpheres.Count; i++)
            {
                var sphereData = activeSpheres[i];
                var materialData = sphereData.Material;
                sphereBuffer[i] = new Sphere(sphereData.Center, sphereData.Radius,
                    new Material(materialData.Type, materialData.Albedo.ToFloat3(), materialData.Fuzz,
                        materialData.RefractiveIndex));
            }
        }        
    }
}