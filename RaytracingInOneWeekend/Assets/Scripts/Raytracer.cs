using Unity.Burst;
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
        [SerializeField] UnityEngine.Camera targetCamera = null;

        [SerializeField] [Range(0.01f, 1)] float resolutionScaling = 1;
        [SerializeField] [Range(1, 1000)] int samplesPerPixel = 100;
        [SerializeField] [Range(1, 100)] int traceDepth = 50;

        CommandBuffer commandBuffer;
        Texture2D backBufferTexture;
        NativeArray<half4> backBuffer;
        NativeArray<Sphere> spheres;

        JobHandle raytraceJobHandle;

        int Width => Mathf.RoundToInt(targetCamera.pixelWidth * resolutionScaling);
        int Height => Mathf.RoundToInt(targetCamera.pixelHeight * resolutionScaling);

        void OnValidate()
        {
            if (Application.isPlaying)
                RebuildBuffers();
        }

        void RebuildBuffers()
        {
            if (backBufferTexture != null)
                Destroy(backBufferTexture);

            backBufferTexture = new Texture2D(Width, Height, TextureFormat.RGBAHalf, false, true)
            {
                filterMode = FilterMode.Point,
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

        void Awake()
        {
            RebuildBuffers();

            spheres = new NativeArray<Sphere>(5, Allocator.Persistent)
            {
                [0] = new Sphere(float3(0, 0, -1), 0.5f, Material.Lambertian(float3(0.1f, 0.2f, 0.5f))),
                [1] = new Sphere(float3(0, -100.5f, -1), 100, Material.Lambertian(float3(0.8f, 0.8f, 0.0f))),
                [2] = new Sphere(float3(1, 0, -1), 0.5f, Material.Metal(float3(0.8f, 0.6f, 0.2f), 0.3f)),
                [3] = new Sphere(float3(-1, 0, -1), 0.5f, Material.Dielectric(1.5f)),
                [4] = new Sphere(float3(-1, 0, -1), -0.45f, Material.Dielectric(1.5f))
            };
        }

        void OnDestroy()
        {
            if (backBuffer.IsCreated) backBuffer.Dispose();
            if (spheres.IsCreated) spheres.Dispose();
        }

        void Update()
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
                Spheres = spheres,
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

    [BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
    struct RaytraceJob : IJobParallelFor
    {
        [ReadOnly] public int2 Size;
        [ReadOnly] public Camera Camera;
        [ReadOnly] public NativeArray<Sphere> Spheres;
        [ReadOnly] public int SampleCount;
        [ReadOnly] public int TraceDepth;
        [ReadOnly] public Random Rng;

        [WriteOnly] public NativeArray<half4> Target;

        bool Color(Ray r, int depth, out float3 color)
        {
            if (Spheres.Hit(r, 0.001f, float.PositiveInfinity, out HitRecord rec))
            {
                if (depth < TraceDepth && rec.Material.Scatter(r, rec, Rng, out float3 attenuation, out Ray scattered))
                {
                    if (Color(scattered, depth + 1, out float3 scatteredColor))
                    {
                        color = attenuation * scatteredColor;
                        return true;
                    }
                }
                color = default;
                return false;
            }

            float3 unitDirection = normalize(r.Direction);
            float t = 0.5f * (unitDirection.y + 1);
            color = lerp(1, float3(0.5f, 0.7f, 1), t);
            return true;
        }

        public void Execute(int index)
        {
            int2 coordinates = int2(
                index % Size.x, // column 
                index / Size.x // row
            );

            float3 colorAcc = 0;
            int realSampleCount = 0;
            for (int s = 0; s < SampleCount; s++)
            {
                float2 normalizedCoordinates = (coordinates + Rng.NextFloat2()) / Size; // (u, v)
                Ray r = Camera.GetRay(normalizedCoordinates);
                if (Color(r, 0, out float3 sampleColor))
                {
                    colorAcc += sampleColor;
                    realSampleCount++;
                }
            }
            float3 finalColor = colorAcc / realSampleCount;

            finalColor = finalColor.LinearToGamma();

            Target[index] = half4(half3(finalColor), half(1));
        }
    }
}