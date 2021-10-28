Shader "Raytracing/Dielectric"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _Glossiness ("Smoothness", Range(0,1)) = 1
        _RefractiveIndex ("Refractive Index", Range(1, 5)) = 1.5
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard alpha finalcolor:tint
        #pragma target 3.0

        sampler2D _MainTex;

        struct Input
        {
            float2 uv_MainTex;
        };

        half _Glossiness, _RefractiveIndex;
        fixed4 _Color;

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            // Albedo comes from a texture tinted by color
            fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
            c.a = 0;
            o.Albedo = c.rgb;
            // Metallic and smoothness come from slider variables
            o.Metallic = 0;
            o.Smoothness = _Glossiness;
            o.Alpha = c.a;
        }

          void tint (Input IN, SurfaceOutputStandard o, inout fixed4 color)
          {
              color.rgb *= _Color.rgb;
          }
        ENDCG
    }
    FallBack "Diffuse"
}
