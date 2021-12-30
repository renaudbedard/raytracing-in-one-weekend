using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Util
{
	enum ChannelType
	{
		UnsignedByte,
		SignedHalf
	}

	[StructLayout(LayoutKind.Explicit)]
	struct RGBA32
	{
		[FieldOffset(0)] public byte r;
		[FieldOffset(1)] public byte g;
		[FieldOffset(2)] public byte b;
		[FieldOffset(3)] public byte a;

		public override string ToString() => $"{{{r}, {g}, {b}, {a}}}";
	}

	[StructLayout(LayoutKind.Explicit)]
	struct RGB24
	{
		[FieldOffset(0)] public byte r;
		[FieldOffset(1)] public byte g;
		[FieldOffset(2)] public byte b;

		public override string ToString() => $"{{{r}, {g}, {b}}}";
	}

	[StructLayout(LayoutKind.Explicit)]
	struct RGBX64
	{
		[FieldOffset(0)] public half r;
		[FieldOffset(2)] public half g;
		[FieldOffset(4)] public half b;
		[FieldOffset(6)] public half _;

		public override string ToString() => $"{{{r}, {g}, {b}}}";
	}
}