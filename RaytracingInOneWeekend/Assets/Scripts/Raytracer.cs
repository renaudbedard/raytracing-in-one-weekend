using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;
using float3 = Unity.Mathematics.float3;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
    public class Raytracer : MonoBehaviour
    {
        [SerializeField] UnityEngine.Camera targetCamera = null;

        CommandBuffer commandBuffer;
        Texture2D backBufferTexture;
        NativeArray<half4> backBuffer;
        NativeArray<Sphere> spheres;

        JobHandle raytraceJobHandle;

        void Awake()
        {
            int width = targetCamera.pixelWidth;
            int height = targetCamera.pixelHeight;

            backBufferTexture = new Texture2D(width, height, TextureFormat.RGBAHalf, false, false);

            commandBuffer = new CommandBuffer { name = "Raytracer" };
            commandBuffer.Blit(backBufferTexture, new RenderTargetIdentifier(BuiltinRenderTextureType.CameraTarget));
            targetCamera.AddCommandBuffer(CameraEvent.AfterEverything, commandBuffer);

            backBuffer = new NativeArray<half4>(width * height,
                Allocator.Persistent,
                NativeArrayOptions.UninitializedMemory);

            spheres = new NativeArray<Sphere>(2, Allocator.Persistent)
            {
                [0] = new Sphere(float3(0, 0, -1), 0.5f),
                [1] = new Sphere(float3(0, -100.5f, -1), 100)
            };
        }

        void OnDestroy()
        {
            if (backBuffer.IsCreated) backBuffer.Dispose();
            if (spheres.IsCreated) spheres.Dispose();
        }

        void Update()
        {
            int width = targetCamera.pixelWidth;
            int height = targetCamera.pixelHeight;
            float aspect = (float) width / height;

            var raytracingCamera = new Camera(0, 
                float3(-aspect, -1, -1), 
                float3(aspect * 2, 0, 0), 
                float3(0, 2, 0));

            var raytraceJob = new RaytraceJob
            {
                Size = int2(width, height), 
                Camera = raytracingCamera, 
                Target = backBuffer,
                Spheres = spheres,
                Rng = new Random(1),
                SampleCount = 100
            };
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
        [ReadOnly] public int2 Size;
        [ReadOnly] public Camera Camera;
        [ReadOnly] public NativeArray<Sphere> Spheres;
        [ReadOnly] public int SampleCount;
        [ReadOnly] public Random Rng;        

        [WriteOnly] public NativeArray<half4> Target;

        float3 Color(Ray r)
        {
            if (Spheres.Hit(r, 0, float.PositiveInfinity, out HitRecord rec))
                return 0.5f * (rec.Normal + 1);

            float3 unitDirection = normalize(r.Direction);
            float t = 0.5f * (unitDirection.y + 1);
            return lerp(1, float3(0.5f, 0.7f, 1), t);
        }

        public void Execute(int index)
        {
            int2 coordinates = int2(
            index % Size.x, // column 
            index / Size.x  // row
            );

            float3 color = 0;
            for (int s = 0; s < SampleCount; s++)
            {
                float2 normalizedCoordinates = (coordinates + Rng.NextFloat2()) / Size; // (u, v)
                Ray r = Camera.GetRay(normalizedCoordinates);
                color += Color(r);
            }
            color /= SampleCount;

            Target[index] = half4(half3(color), half(1));
        }
    }

    struct Ray
    {
        public readonly float3 Origin;
        public readonly float3 Direction;

        public Ray(float3 origin, float3 direction)
        {
            Origin = origin;
            Direction = direction;
        }

        public float3 GetPoint(float t) => Origin + t * Direction;
    }

    struct HitRecord
    {
        public readonly float Distance;
        public readonly float3 Point;
        public readonly float3 Normal;

        public HitRecord(float distance, float3 point, float3 normal)
        {
            Distance = distance;
            Point = point;
            Normal = normal;
        }
    }

    struct Sphere
    {
        public readonly float3 Center;
        public readonly float Radius;

        public Sphere(float3 center, float radius)
        {
            Center = center;
            Radius = radius;
        }

        public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
        {
            float3 oc = r.Origin - Center;
            float a = dot(r.Direction, r.Direction);
            float b = dot(oc, r.Direction);
            float c = dot(oc, oc) - Radius * Radius;
            float discriminant = b * b - a * c;

            if (discriminant > 0)
            {
                float sqrtDiscriminant = sqrt(discriminant);
                float t = (-b - sqrtDiscriminant) / a;
                if (t < tMax && t > tMin)
                {
                    float3 point = r.GetPoint(t);
                    rec = new HitRecord(t, point, (point - Center) / Radius);
                    return true;
                }

                t = (-b + sqrtDiscriminant) / a;
                if (t < tMax && t > tMin)
                {
                    float3 point = r.GetPoint(t);
                    rec = new HitRecord(t, point, (point - Center) / Radius);
                    return true;
                }
            }

            rec = default;
            return false;
        }
    }

    static class HittableExtensions
    {
        public static bool Hit(this NativeArray<Sphere> spheres, Ray r, float tMin, float tMax, out HitRecord rec)
        {
            bool hitAnything = false;
            rec = new HitRecord(tMax, 0, 0);

            for (var i = 0; i < spheres.Length; i++)
            {
                Sphere sphere = spheres[i];
                if (sphere.Hit(r, tMin, tMax, out HitRecord thisRec) && thisRec.Distance < rec.Distance)
                {
                    hitAnything = true;
                    rec = thisRec;
                }
            }

            return hitAnything;
        }
    }

    struct Camera
    {
        public readonly float3 Origin;
        public readonly float3 LowerLeftCorner;
        public readonly float3 Horizontal;
        public readonly float3 Vertical;

        public Camera(float3 origin, float3 lowerLeftCorner, float3 horizontal, float3 vertical)
        {
            Origin = origin;
            LowerLeftCorner = lowerLeftCorner;
            Horizontal = horizontal;
            Vertical = vertical;
        }

        public Ray GetRay(float2 normalizedCoordinates) => new Ray(Origin,
                LowerLeftCorner + normalizedCoordinates.x * Horizontal + normalizedCoordinates.y * Vertical - Origin);
    }
}