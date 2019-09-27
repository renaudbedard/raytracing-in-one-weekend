using System;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;

namespace OptiX
{
	public struct SizeT
	{
#if UNITY_64
		private ulong Value;
		public static implicit operator ulong(SizeT operand) => operand.Value;
		public static implicit operator SizeT(ulong operand) => new SizeT { Value = operand };
		public static implicit operator SizeT(uint operand) => new SizeT { Value = operand };
		public static implicit operator SizeT(long operand) => new SizeT { Value = (ulong) operand };
		public static implicit operator SizeT(int operand) => new SizeT { Value = (uint) operand };
#else
		private uint Value;
		public static implicit operator uint(SizeT operand) => operand.Value;
		public static implicit operator SizeT(uint operand) => new SizeT { Value = operand };
#endif
	}

	public enum CudaError
	{
		Success = 0,
		ErrorInvalidValue = 1,
		ErrorMemoryAllocation = 2,
		ErrorInvalidMemcpyDirection = 21,
		ErrorIllegalAddress = 700
	}

	public enum CudaMemcpyKind
	{
		HostToHost = 0,
		HostToDevice = 1,
		DeviceToHost = 2,
		DeviceToDevice = 3,
		Default = 4,
	}

	public enum OptixResult
	{
		Success = 0,
		ErrorInvalidValue = 7001,
		ErrorHostOutOfMemory = 7002,
		ErrorInvalidOperation = 7003,
		ErrorFileIoError = 7004,
		ErrorInvalidFileFormat = 7005,
		ErrorDiskCacheInvalidPath = 7010,
		ErrorDiskCachePermissionError = 7011,
		ErrorDiskCacheDatabaseError = 7012,
		ErrorDiskCacheInvalidData = 7013,
		ErrorLaunchFailure = 7050,
		ErrorInvalidDeviceContext = 7051,
		ErrorCudaNotInitialized = 7052,
		ErrorInvalidPtx = 7200,
		ErrorInvalidLaunchParameter = 7201,
		ErrorInvalidPayloadAccess = 7202,
		ErrorInvalidAttributeAccess = 7203,
		ErrorInvalidFunctionUse = 7204,
		ErrorInvalidFunctionArguments = 7205,
		ErrorPipelineOutOfConstantMemory = 7250,
		ErrorPipelineLinkError = 7251,
		ErrorInternalCompilerError = 7299,
		ErrorDenoiserModelNotSet = 7300,
		ErrorDenoiserNotInitialized = 7301,
		ErrorAccelNotCompatible = 7400,
		ErrorNotSupported = 7800,
		ErrorUnsupportedAbiVersion = 7801,
		ErrorFunctionTableSizeMismatch = 7802,
		ErrorInvalidEntryFunctionOptions = 7803,
		ErrorLibraryNotFound = 7804,
		ErrorEntrySymbolNotFound = 7805,
		ErrorCudaError = 7900,
		ErrorInternalError = 7990,
		ErrorUnknown = 7999,
	}

	public enum OptixLogLevel
	{
		Disable = 0,
		Fatal = 1,
		Error = 2,
		Warning = 3,
		Print = 4
	}

	public enum OptixPixelFormat
	{
		Half3  = 0x2201,
		Half4  = 0x2202,
		Float3 = 0x2203,
		Float4 = 0x2204,
		Uchar3 = 0x2205, // Not implemented
		Uchar4 = 0x2206  // Not implemented
	}

	public enum OptixDenoiserInputKind
	{
		Rgb = 0x2301,
		RgbAlbedo = 0x2302,
		RgbAlbedoNormal = 0x2303, // Not implemented
	}

	public struct OptixDenoiserOptions
	{
		public OptixDenoiserInputKind InputKind;
		public OptixPixelFormat PixelFormat;
	}

	public enum OptixModelKind
	{
		User = 0x2321,
		Ldr = 0x2322,
		Hdr = 0x2323
	}

	public struct OptixDenoiserParams
	{
		public uint DenoiseAlpha;
		public CudaBuffer HdrIntensity;
		public float BlendFactor;
	}

	public struct OptixImage2D
	{
		public CudaBuffer Data;
		public uint Width;
		public uint Height;
		public uint RowStrideInBytes;
		public uint PixelStrideInBytes;
		public OptixPixelFormat Format;
	}

	public struct OptixDenoiserSizes
	{
		public SizeT StateSizeInBytes;
		public SizeT MinimumScratchSizeInBytes;
		public SizeT RecommendedScratchSizeInBytes;
		public uint OverlapWindowSizeInPixels;
	}

	// TODO: IntPtr is actually a const char*
	public delegate void OptixErrorFunction(OptixLogLevel level, string tag, string message, IntPtr cbdata);

	public static class OptixApi
	{
#if UNITY_64
		public const string LibraryFilename = "OptiXDenoiser_win64.dll";
#else
		public const string LibraryFilename = "OptiXDenoiser_win32.dll";
#endif

		[DllImport(LibraryFilename, EntryPoint = "fullTest")]
		public static extern void FullTest(OptixErrorFunction logCallback);
	}

	public struct OptixDenoiser
	{
		[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "createDenoiser")]
		public static extern unsafe OptixResult Create(OptixDeviceContext context, OptixDenoiserOptions* options, ref OptixDenoiser outDenoiser);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "setDenoiserModel")]
		public static extern OptixResult SetModel(OptixDenoiser denoiser, OptixModelKind kind, IntPtr data, SizeT sizeInBytes);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "computeIntensity")]
		public static extern unsafe OptixResult ComputeIntensity(OptixDenoiser denoiser, CudaStream stream,
			OptixImage2D* inputImage, CudaBuffer outputIntensity, CudaBuffer scratch, SizeT scratchSizeInBytes);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "computeMemoryResources")]
		public static extern unsafe OptixResult ComputeMemoryResources(OptixDenoiser denoiser, uint outputWidth, uint outputHeight,
			OptixDenoiserSizes* returnSizes);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "invokeDenoiser")]
		public static extern unsafe OptixResult Invoke(OptixDenoiser denoiser, CudaStream stream,
			OptixDenoiserParams* parameters, CudaBuffer denoiserState, SizeT denoiserStateSizeInBytes,
			OptixImage2D* inputLayers, uint numInputLayers, uint inputOffsetX, uint inputOffsetY,
			OptixImage2D* outputLayer, CudaBuffer scratch, SizeT scratchSizeInBytes);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "destroyDenoiser")]
		public static extern OptixResult Destroy(OptixDenoiser device);
	}

	public struct OptixDeviceContext
	{
		[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "createDeviceContext")]
		public static extern OptixDeviceContext Create(OptixErrorFunction logCallback, OptixLogLevel logLevel);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "destroyDeviceContext")]
		public static extern void Destroy(OptixDeviceContext device);
	}

	public struct CudaStream
	{
		[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "createCudaStream")]
		public static extern CudaError Create(ref CudaStream outStream);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "destroyCudaStream")]
		public static extern CudaError Destroy(CudaStream stream);
	}

	public struct CudaBuffer
	{
		[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "allocateCudaBuffer")]
		public static extern CudaError Allocate(SizeT sizeInBytes, ref CudaBuffer outBuffer);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "deallocateCudaBuffer")]
		public static extern CudaError Deallocate(CudaBuffer buffer);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "copyCudaBuffer")]
		public static extern CudaError Copy(IntPtr source, IntPtr destination, SizeT size, CudaMemcpyKind kind);

		public CudaError EnsureCapacity(SizeT lastSizeInBytes, SizeT newSizeInBytes)
		{
			if (newSizeInBytes <= lastSizeInBytes && Handle != IntPtr.Zero)
				return CudaError.Success;

			if (Handle != IntPtr.Zero)
			{
				CudaError error = Deallocate(this);
				if (error != CudaError.Success) return error;
			}
			return Allocate(newSizeInBytes, ref this);
		}
	}
}