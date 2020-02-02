Shader "Hidden/ViewRange"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Cull Off
        ZWrite Off
        ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            uniform int _Channel;
            uniform float4 _Minimum_Range;
            uniform int _NormalDisplay;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;

            // from https://www.shadertoy.com/view/WlfXRN
            // License CC0 (public domain)
            // https://creativecommons.org/share-your-work/public-domain/cc0/
            float3 inferno(float t)
            {
                const float3 c0 = float3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
                const float3 c1 = float3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
                const float3 c2 = float3(11.60249308247187, -3.972853965665698, -15.9423941062914);
                const float3 c3 = float3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
                const float3 c4 = float3(77.162935699427, -33.40235894210092, -81.80730925738993);
                const float3 c5 = float3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
                const float3 c6 = float3(25.13112622477341, -12.24266895238567, -23.07032500287172);
                return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
            }

            float4 frag(v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);

                if (_NormalDisplay == 1)
                    return float4(GammaToLinearSpace(normalize(col.yzw) * 0.5 + 0.5), 1);

                return float4(inferno(saturate((col[_Channel] - _Minimum_Range.x) / _Minimum_Range.y)), 1);
            }
            ENDCG
        }
    }
}