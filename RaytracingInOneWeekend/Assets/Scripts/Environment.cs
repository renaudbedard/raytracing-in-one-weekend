using Unity.Mathematics;

namespace RaytracerInOneWeekend
{
	enum SkyType
	{
		GradientSky,
		CubeMap
	}

	struct Environment
	{
		public SkyType SkyType;
		public float3 SkyBottomColor;
		public float3 SkyTopColor;
		public Cubemap SkyCubemap;
	}
}