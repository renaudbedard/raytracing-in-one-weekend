using System.Runtime.InteropServices;

namespace RaytracerInOneWeekend
{
	[StructLayout(LayoutKind.Explicit)]
	struct RGBA32
	{
		[FieldOffset(0)] public byte r;
		[FieldOffset(1)] public byte g;
		[FieldOffset(2)] public byte b;
		[FieldOffset(3)] public byte a;
	}
}