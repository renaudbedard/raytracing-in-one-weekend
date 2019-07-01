using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;
using float3 = Unity.Mathematics.float3;

public class Raytracer : MonoBehaviour
{
    [SerializeField] Camera targetCamera = null;

    CommandBuffer commandBuffer;
    Texture2D backBufferTexture;
    NativeArray<half4> backBuffer;

    JobHandle raytraceJobHandle;

    void Awake()
    {
        int width = targetCamera.pixelWidth;
        int height = targetCamera.pixelHeight;

        backBufferTexture = new Texture2D(width, height, TextureFormat.RGBAHalf, false, false);
        
        commandBuffer = new CommandBuffer { name = "Raytracer" };
        commandBuffer.Blit(backBufferTexture, new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget));
        targetCamera.AddCommandBuffer(CameraEvent.AfterEverything, commandBuffer);        

        backBuffer = new NativeArray<half4>(
            width * height,
                Allocator.Persistent, 
                NativeArrayOptions.UninitializedMemory);
    }

    void OnDestroy()
    {
        if (backBuffer.IsCreated)
            backBuffer.Dispose();
    }

    void Update()
    {
        int width = targetCamera.pixelWidth;
        int height = targetCamera.pixelHeight;
        
        var raytraceJob = new RaytraceJob { Width = width, Height = height, Target = backBuffer };
        raytraceJobHandle = raytraceJob.Schedule(width * height, width);
    }

    void LateUpdate()
    {
        raytraceJobHandle.Complete();
        
        backBufferTexture.LoadRawTextureData(backBuffer);
        backBufferTexture.Apply(false);
    }
}

[BurstCompile]
struct RaytraceJob : IJobParallelFor
{
    [ReadOnly] public int Width;
    [ReadOnly] public int Height;
        
    [WriteOnly] public NativeArray<half4> Target;

    float HitSphere(float3 center, float radius, Ray r)
    {
        float3 oc = r.Origin - center;
        float a = dot(r.Direction, r.Direction);
        float b = 2 * dot(oc, r.Direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0)
            return -1;
        return (-b - sqrt(discriminant)) / 2 * a;
    }

    float3 Color(Ray r)
    {
        float t = HitSphere(float3(0, 0, -1), 0.5f, r);
        if (t > 0)
        {
            float3 n = normalize(r.GetPoint(t) - float3(0, 0, -1));
            return 0.5f * (n + float3(1, 1, 1));
        }
        
        float3 unitDirection = normalize(r.Direction);
        t = 0.5f * (unitDirection.y + 1);
        return lerp(float3(1, 1, 1), float3(0.5f, 0.7f, 1), t);
    }

    public void Execute(int index)
    {
        float aspect = (float) Width / Height;
        
        float3 lowerLeftCorner = float3(-aspect, -1, -1);
        float3 horizontal = float3(aspect * 2, 0, 0);
        float3 vertical = float3(0, 2, 0);
        float3 origin = float3(0, 0, 0);

        int row = index / Width;
        int col = index % Width;
        
        float u = (float) col / Width;
        float v = (float) row / Height;
        
        var r = new Ray(origin, lowerLeftCorner + u * horizontal + v * vertical);
        float3 color = Color(r); 
        Target[row * Width + col] = half4(half3(color), half(1));
    }
}

struct Ray
{
    public float3 Origin;
    public float3 Direction;

    public Ray(float3 origin, float3 direction)
    {
        Origin = origin;
        Direction = direction;
    }

    public float3 GetPoint(float t) => Origin + t * Direction;
}