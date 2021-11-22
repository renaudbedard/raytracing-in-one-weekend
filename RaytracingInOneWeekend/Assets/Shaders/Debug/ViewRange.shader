Shader "Hidden/ViewRange"
{
    Properties
    {
        _MainTex("Main Texture", 2D) = "grey" {}
    }

    HLSLINCLUDE

    #pragma target 4.5
    #pragma only_renderers d3d11 vulkan metal switch

    #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
    #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"
    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/ShaderLibrary/ShaderVariables.hlsl"

    struct Attributes
    {
        uint vertexID : SV_VertexID;
        UNITY_VERTEX_INPUT_INSTANCE_ID
    };

    struct Varyings
    {
        float4 positionCS : SV_POSITION;
        float2 texcoord   : TEXCOORD0;
        UNITY_VERTEX_OUTPUT_STEREO

    };

    Varyings Vert(Attributes input)
    {
        Varyings output;

        UNITY_SETUP_INSTANCE_ID(input);
        UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

        output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
        output.texcoord = GetFullScreenTriangleTexCoord(input.vertexID);

        return output;
    }

    // List of properties to control your post process effect
    TEXTURE2D(_MainTex);
    int _Channel;
    float4 _Minimum_Range;
    int _NormalDisplay;

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

    float4 Blit(Varyings input) : SV_Target
    {
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

#if UNITY_UV_STARTS_AT_TOP
        input.texcoord.y = 1 - input.texcoord.y;
#endif

        float4 sourceColor = SAMPLE_TEXTURE2D(_MainTex, s_point_clamp_sampler, input.texcoord);

        if (_NormalDisplay == 1)
            return float4(LinearToSRGB(normalize(sourceColor.yzw) * 0.5f + 0.5f), 1);

        float value = saturate((sourceColor[_Channel] - _Minimum_Range.x) / _Minimum_Range.y);
        return float4(inferno(value), 1);
    }

    ENDHLSL

    SubShader
    {
        Pass
        {
            Name "Blit"

            ZWrite Off
            ZTest Always
            Blend Off
            Cull Off

            HLSLPROGRAM
                #pragma fragment Blit
                #pragma vertex Vert
            ENDHLSL
        }
    }

    Fallback Off
}