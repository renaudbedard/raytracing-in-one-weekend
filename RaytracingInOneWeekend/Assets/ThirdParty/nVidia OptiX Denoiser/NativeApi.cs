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
	public static class NativeApi
	{
#if UNITY_64
		const string LibraryFilename = "OptiXDenoiser_win64.dll";
#else
		const string LibraryFilename = "OptiXDenoiser_win32.dll";
#endif

		// TODO: IntPtr is actually a const char*
		public delegate void ErrorFunction(uint level, string tag, string message, IntPtr cbdata);

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

		public struct Denoiser
		{
			[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

			public enum InputKind
			{
				Rgb = 0x2301,
				RgbAlbedo = 0x2302,
				RgbAlbedoNormal = 0x2303,
			}

			public struct Options
			{
				public InputKind InputKind;
				public OptixPixelFormat PixelFormat;
			}

			[DllImport(LibraryFilename, EntryPoint = "createDenoiser")]
			public static extern unsafe OptixResult Create(DeviceContext context, Options* options, ref Denoiser denoiser);

			[DllImport(LibraryFilename, EntryPoint = "destroyDenoiser")]
			public static extern OptixResult Destroy(Denoiser device);
		}

		public struct DeviceContext
		{
			[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

			[DllImport(LibraryFilename, EntryPoint = "createContext")]
			public static extern DeviceContext Create(ErrorFunction logCallback, int logLevel);

			[DllImport(LibraryFilename, EntryPoint = "destroyContext")]
			public static extern void Destroy(DeviceContext device);
		}
	}
}