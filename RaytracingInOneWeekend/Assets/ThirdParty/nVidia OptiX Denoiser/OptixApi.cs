using System;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;

#if UNITY_64
using SizeT = System.UInt64;
#else
using SizeT = System.UInt32;
#endif

namespace OptiX
{
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

	public enum OptixPixelFormat
	{
		Half3  = 0x2201,
		Half4  = 0x2202,
		Float3 = 0x2203,
		Float4 = 0x2204,
		Uchar3 = 0x2205,
		Uchar4 = 0x2206
	}

	public enum OptixDenoiserInputKind
	{
		Rgb = 0x2301,
		RgbAlbedo = 0x2302,
		RgbAlbedoNormal = 0x2303,
	}

	public struct OptixDenoiserOptions
	{
		public OptixDenoiserInputKind InputKind;
		public OptixPixelFormat PixelFormat;
	}

	// TODO: IntPtr is actually a const char*
	public delegate void OptixErrorFunction(uint level, string tag, string message, IntPtr cbdata);

	static class OptixApi
	{
#if UNITY_64
		public const string LibraryFilename = "OptiXDenoiser_win64.dll";
#else
		public const string LibraryFilename = "OptiXDenoiser_win32.dll";
#endif
	}

	public struct OptixDenoiser
	{
		[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "createDenoiser")]
		public static extern unsafe OptixResult Create(OptixDeviceContext context, OptixDenoiserOptions* options, ref OptixDenoiser denoiser);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "destroyDenoiser")]
		public static extern OptixResult Destroy(OptixDenoiser device);
	}

	public struct OptixDeviceContext
	{
		[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "createContext")]
		public static extern OptixDeviceContext Create(OptixErrorFunction logCallback, int logLevel);

		[DllImport(OptixApi.LibraryFilename, EntryPoint = "destroyContext")]
		public static extern void Destroy(OptixDeviceContext device);
	}
}