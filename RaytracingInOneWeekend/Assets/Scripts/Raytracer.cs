using System.Linq;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
    public class Raytracer : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] UnityEngine.Camera targetCamera = null;

        [Header("Settings")]
        [SerializeField] [Range(0.01f, 2)] float resolutionScaling = 1;
        [SerializeField] [Range(1, 1000)] int samplesPerPixel = 100;
        [SerializeField] [Range(1, 100)] int traceDepth = 50;

        [Header("World")] [SerializeField] SphereData[] spheres = null;
        
        CommandBuffer commandBuffer;
        Texture2D backBufferTexture;
        NativeArray<half4> backBuffer;
        NativeArray<Sphere> sphereBuffer;

        JobHandle raytraceJobHandle;

        int Width => Mathf.RoundToInt(targetCamera.pixelWidth * resolutionScaling);
        int Height => Mathf.RoundToInt(targetCamera.pixelHeight * resolutionScaling);

        void OnValidate()
        {
            if (Application.isPlaying)
            {
                RebuildBuffers();
                RebuildWorld();
            }
        }
        
        void Awake()
        {
            RebuildBuffers();
            RebuildWorld();
        }
        
        void OnDestroy()
        {
            if (backBuffer.IsCreated) backBuffer.Dispose();
            if (sphereBuffer.IsCreated) sphereBuffer.Dispose();
        }
        
        void RebuildBuffers()
        {
            if (backBufferTexture != null)
                Destroy(backBufferTexture);

            backBufferTexture = new Texture2D(Width, Height, TextureFormat.RGBAHalf, false, true)
            {
                filterMode = resolutionScaling > 1 ? FilterMode.Bilinear : FilterMode.Point,
                hideFlags = HideFlags.HideAndDontSave
            };

            if (commandBuffer != null)
            {
                targetCamera.RemoveCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
                commandBuffer.Release();
            }

            commandBuffer = new CommandBuffer { name = "Raytracer" };
            commandBuffer.Blit(backBufferTexture, new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget));
            targetCamera.AddCommandBuffer(CameraEvent.AfterEverything, commandBuffer);

            if (backBuffer.IsCreated)
                backBuffer.Dispose();

            backBuffer = new NativeArray<half4>(Width * Height,
                Allocator.Persistent,
                NativeArrayOptions.UninitializedMemory);
        }

        void RebuildWorld()
        {
            if (sphereBuffer.IsCreated)
                sphereBuffer.Dispose();

            sphereBuffer = new NativeArray<Sphere>(spheres.Count(x => x.Enabled), Allocator.Persistent);

            int i = 0;
            foreach (var sphere in spheres)
            {
                if (!sphere.Enabled)
                    continue;
                
                var materialData = sphere.Material;
                sphereBuffer[i++] = new Sphere(sphere.Center, sphere.Radius,
                    new Material(materialData.Type, materialData.Albedo.ToFloat3(), materialData.Fuzz,
                        materialData.RefractiveIndex));
            }
        }

        void Update()
        {
#if UNITY_EDITOR
            if (spheres.Any(x => x.Material.Dirty))
            {
                foreach (var sphere in spheres) sphere.Material.Dirty = false;
                RebuildWorld();
            }
#endif
            
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
                Rng = new Random((uint) Time.frameCount),
                SampleCount = samplesPerPixel,
                TraceDepth = traceDepth
            };
            raytraceJobHandle = raytraceJob.Schedule(Width * Height, Width);
        }

        void LateUpdate()
        {
            raytraceJobHandle.Complete();

            backBufferTexture.LoadRawTextureData(backBuffer);
            backBufferTexture.Apply(false);
        }
    }
}