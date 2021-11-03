Shader "Hidden/Blit"
{
    Properties
    {
        _MainTex("Main Texture", 2D) = "grey" {}
    }

    HLSLINCLUDE

    #pragma target 4.5
    #pragma only_renderers d3d11 vulkan metal switch

    #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
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

    float4 Blit(Varyings input) : SV_Target
    {
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

#if UNITY_UV_STARTS_AT_TOP
            input.texcoord.y = 1 - input.texcoord.y;
#endif

        float3 sourceColor = SAMPLE_TEXTURE2D(_MainTex, s_point_clamp_sampler, input.texcoord).rgb;
        return float4(sourceColor, 1);
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