// NOTE: library only exists in 64-bit variant

#if UNITY_64
using System;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using SizeT = System.UInt64;

namespace OpenImageDenoise
{
	public static class OidnApi
	{
		public const string LibraryFilename = "OpenImageDenoise.dll";
	}

	/// <summary>
	/// Error codes
	/// </summary>
	public enum OidnError
	{
		/// <summary>
		/// No error occurred
		/// </summary>
		None = 0,

		/// <summary>
		/// An unknown error occurred
		/// </summary>
		Unknown = 1,

		/// <summary>
		/// An invalid argument was specified
		/// </summary>
		InvalidArgument = 2,

		/// <summary>
		/// The operation is not allowed
		/// </summary>
		InvalidOperation = 3,

		/// <summary>
		/// Not enough memory to execute the operation
		/// </summary>
		OutOfMemory = 4,

		/// <summary>
		/// The hardware (e.g. CPU) is not supported
		/// </summary>
		UnsupportedHardware = 5,

		/// <summary>
		/// The operation was cancelled by the user
		/// </summary>
		Cancelled = 6
	}

	/// <summary>
	/// Error callback function
	/// </summary>
	public delegate void OidnErrorFunction(IntPtr userPtr, OidnError code, string message);

	public struct OidnDevice
	{
		[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

		/// <summary>
		/// Open Image Denoise device types
		/// </summary>
		public enum Type
		{
			/// <summary>
			/// Select device automatically
			/// </summary>
			Default = 0,

			/// <summary>
			/// CPU device
			/// </summary>
			Cpu = 1
		}

		/// <summary>
		/// Creates a new Open Image Denoise device.
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnNewDevice")]
		public static extern OidnDevice New(Type type);

		/// <summary>
		/// Commits all previous changes to the device.
		/// Must be called before first using the device (e.g. creating filters).
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnCommitDevice")]
		public static extern void Commit(OidnDevice device);

		/// <summary>
		/// Releases the device (decrements the reference count).
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnReleaseDevice")]
		public static extern void Release(OidnDevice device);

		/// <summary>
		/// Sets the error callback function of the device.
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnSetDeviceErrorFunction")]
		public static extern void SetErrorFunction(OidnDevice device, OidnErrorFunction func, IntPtr userPtr);
	}

	public struct OidnBuffer
	{
		/// <summary>
		/// Formats for images and other data stored in buffers
		/// </summary>
		public enum Format
		{
			Undefined = 0,

			// 32-bit single-precision floating point scalar and vector formats
			Float = 1,
			Float2 = 2,
			Float3 = 3,
			Float4 = 4,
		}
	}

	public struct OidnFilter
	{
		[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

		/// <summary>
		/// Creates a new filter of the specified type (e.g. "RT").
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnNewFilter")]
		public static extern OidnFilter New(OidnDevice device, string type);

		/// <summary>
		/// Sets an image parameter of the filter (owned by the user).
		/// If bytePixelStride and/or byteRowStride are zero, these will be computed automatically.
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnSetSharedFilterImage")]
		public static extern OidnDevice SetSharedImage(OidnFilter oidnFilter, string name, IntPtr ptr,
			OidnBuffer.Format format, SizeT width, SizeT height, SizeT byteOffset,
			SizeT bytePixelStride, SizeT byteRowStride);

		/// <summary>
		/// Sets a boolean parameter of the filter.
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnSetFilter1b")]
		public static extern OidnDevice Set(OidnFilter oidnFilter, string name, bool value);

		/// <summary>
		/// Commits all previous changes to the filter.
		/// Must be called before first executing the filter.
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnCommitFilter")]
		public static extern OidnDevice Commit(OidnFilter oidnFilter);

		/// <summary>
		/// Executes the filter.
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnExecuteFilter")]
		public static extern OidnDevice Execute(OidnFilter oidnFilter);

		/// <summary>
		/// Releases the filter (decrements the reference count).
		/// </summary>
		[DllImport(OidnApi.LibraryFilename, EntryPoint = "oidnReleaseFilter")]
		public static extern OidnDevice Release(OidnFilter oidnFilter);
	}
}

#endif