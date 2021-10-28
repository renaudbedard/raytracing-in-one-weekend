Shader "Raytracing/Probabilistic Volume (2nd Pass)"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _Density ("Density", Range(0,10)) = 0.1
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        CGPROGRAM
        #pragma surface surf Standard vertex:vert alpha
        #pragma target 3.0

        float4 _Color;
        float _Density;
        sampler2D _CameraDepthTexture;

        struct Input
        {
            float4 screenPos;
            float eyeDepth : TEXCOORD4;
        };

        void vert (inout appdata_full v, out Input o)
        {
            UNITY_INITIALIZE_OUTPUT(Input, o);
            COMPUTE_EYEDEPTH(o.eyeDepth);
            v.normal = normalize(v.vertex);
        }

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            o.Albedo = _Color.rgb;
            o.Metallic = 0;
            o.Smoothness = 0;

            float depth = tex2Dproj(_CameraDepthTexture, UNITY_PROJ_COORD(IN.screenPos));
            depth = LinearEyeDepth(depth);
            float distanceInVolume = depth - IN.eyeDepth;

            o.Alpha = saturate(distanceInVolume * _Density);
        }
        ENDCG
    }
}
