using Unity.Mathematics;

namespace Runtime
{
	enum SkyType
	{
		None,
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