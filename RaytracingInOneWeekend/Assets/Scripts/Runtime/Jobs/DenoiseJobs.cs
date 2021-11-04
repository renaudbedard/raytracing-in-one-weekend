using System;
using OpenImageDenoise;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
namespace Runtime.Jobs
{
	// Because the OIDN API uses strings, we can't use Burst here
	struct OpenImageDenoiseJob : IJob
	{
		[ReadOnly] public NativeReference<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> InputColor, InputNormal, InputAlbedo;
		[ReadOnly] public ulong Width, Height;

		public NativeArray<float3> OutputColor;

		public OidnFilter DenoiseFilter;

		public unsafe void Execute()
		{
			if (CancellationToken.Value)
				return;

			OidnFilter.SetSharedImage(DenoiseFilter, "color", new IntPtr(InputColor.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);
			OidnFilter.SetSharedImage(DenoiseFilter, "normal", new IntPtr(InputNormal.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);
			OidnFilter.SetSharedImage(DenoiseFilter, "albedo", new IntPtr(InputAlbedo.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);

			OidnFilter.SetSharedImage(DenoiseFilter, "output", new IntPtr(OutputColor.GetUnsafePtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);

			OidnFilter.Commit(DenoiseFilter);
			OidnFilter.Execute(DenoiseFilter);
		}
	}

#if ENABLE_OPTIX
	// Disabled because it current won't compile using Burst (I swear it used to work)
	// [BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct OptixDenoiseJob : IJob
	{
		[ReadOnly] public NativeReference<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> InputColor, InputAlbedo;
		[ReadOnly] public uint2 BufferSize;
		[ReadOnly] public OptixDenoiserSizes DenoiserSizes;

		[WriteOnly] public NativeArray<float3> OutputColor;

		public OptixDenoiser Denoiser;
		public CudaStream CudaStream;

		public CudaBuffer InputColorBuffer,
			InputAlbedoBuffer,
			OutputColorBuffer,
			ScratchMemory,
			DenoiserState;

		[BurstDiscard]
		static void Check(CudaError cudaError)
		{
			if (cudaError != CudaError.Success)
				Debug.LogError($"CUDA Error : {cudaError}");
		}

		public unsafe void Execute()
		{
			if (CancellationToken.Value)
				return;

			Check(CudaBuffer.Copy(new IntPtr(InputColor.GetUnsafeReadOnlyPtr()), InputColorBuffer.Handle,
				InputColor.Length * sizeof(float3), CudaMemcpyKind.HostToDevice));
			Check(CudaBuffer.Copy(new IntPtr(InputAlbedo.GetUnsafeReadOnlyPtr()), InputAlbedoBuffer.Handle,
				InputAlbedo.Length * sizeof(float3), CudaMemcpyKind.HostToDevice));

			var colorImage = new OptixImage2D
			{
				Data = InputColorBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};
			var albedoImage = new OptixImage2D
			{
				Data = InputAlbedoBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};

			OptixImage2D* optixImages = stackalloc OptixImage2D[2];
			optixImages[0] = colorImage;
			optixImages[1] = albedoImage;

			OptixDenoiserParams denoiserParams = default;

			var outputImage = new OptixImage2D
			{
				Data = OutputColorBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};

			OptixDenoiser.Invoke(Denoiser, CudaStream, &denoiserParams, DenoiserState, DenoiserSizes.StateSizeInBytes,
				optixImages, 2, 0, 0,
				&outputImage, ScratchMemory, DenoiserSizes.RecommendedScratchSizeInBytes);

			Check(CudaBuffer.Copy(OutputColorBuffer.Handle, new IntPtr(OutputColor.GetUnsafePtr()),
				OutputColor.Length * sizeof(float3), CudaMemcpyKind.DeviceToHost));
		}
	}
#endif
}