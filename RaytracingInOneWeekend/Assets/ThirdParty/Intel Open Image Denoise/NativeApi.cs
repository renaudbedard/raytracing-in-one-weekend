// library only exists in 64-bit variant

#if UNITY_64
using System;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using SizeT = System.UInt64;

namespace OpenImageDenoise
{
	public static class NativeApi
	{
		const string LibraryFilename = "OpenImageDenoise.dll";

		/// <summary>
		/// Error codes
		/// </summary>
		public enum Error
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
		public delegate void ErrorFunction(IntPtr userPtr, Error code, string message);

		public struct Device
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
			[DllImport(LibraryFilename, EntryPoint = "oidnNewDevice")]
			public static extern Device New(Type type);

			/// <summary>
			/// Commits all previous changes to the device.
			/// Must be called before first using the device (e.g. creating filters).
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnCommitDevice")]
			public static extern void Commit(Device device);

			/// <summary>
			/// Releases the device (decrements the reference count).
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnReleaseDevice")]
			public static extern void Release(Device device);

			/// <summary>
			/// Sets the error callback function of the device.
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnSetDeviceErrorFunction")]
			public static extern void SetErrorFunction(Device device, ErrorFunction func, IntPtr userPtr);
		}

		public struct Buffer
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

		public struct Filter
		{
			[NativeDisableUnsafePtrRestriction] public IntPtr Handle;

			/// <summary>
			/// Creates a new filter of the specified type (e.g. "RT").
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnNewFilter")]
			public static extern Filter New(Device device, string type);

			/// <summary>
			/// Sets an image parameter of the filter (owned by the user).
			/// If bytePixelStride and/or byteRowStride are zero, these will be computed automatically.
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnSetSharedFilterImage")]
			public static extern Device SetSharedImage(Filter filter, string name, IntPtr ptr,
				Buffer.Format format, SizeT width, SizeT height, SizeT byteOffset,
				SizeT bytePixelStride, SizeT byteRowStride);

			/// <summary>
			/// Sets a boolean parameter of the filter.
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnSetFilter1b")]
			public static extern Device Set(Filter filter, string name, bool value);

			/// <summary>
			/// Commits all previous changes to the filter.
			/// Must be called before first executing the filter.
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnCommitFilter")]
			public static extern Device Commit(Filter filter);

			/// <summary>
			/// Executes the filter.
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnExecuteFilter")]
			public static extern Device Execute(Filter filter);

			/// <summary>
			/// Releases the filter (decrements the reference count).
			/// </summary>
			[DllImport(LibraryFilename, EntryPoint = "oidnReleaseFilter")]
			public static extern Device Release(Filter filter);
		}
	}
}

#endif