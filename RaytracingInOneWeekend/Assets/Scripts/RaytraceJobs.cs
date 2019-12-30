using System;
using System.Runtime.InteropServices;
using OpenImageDenoise;
using OptiX;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using Debug = UnityEngine.Debug;
using Random = Unity.Mathematics.Random;

namespace RaytracerInOneWeekend
{
	struct Diagnostics
	{
#if FULL_DIAGNOSTICS && BVH_ITERATIVE
		public float RayCount;
		public float BoundsHitCount;
		public float CandidateCount;
#pragma warning disable 649
		public float Padding;
#pragma warning restore 649
#else
		public float RayCount;
#endif
	}

#if PATH_DEBUGGING
	struct DebugPath
	{
		public float3 From, To;
	}
#endif

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	unsafe struct AccumulateJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

#if BVH_ITERATIVE
		public struct WorkingArea
		{
			public BvhNode** Nodes;
			public Entity* Entities;
		}
#endif
		[ReadOnly] public float2 Size;
		[ReadOnly] public int SliceOffset;
		[ReadOnly] public int SliceDivider;
		[ReadOnly] public uint Seed;
		[ReadOnly] public Camera Camera;
		[ReadOnly] public uint SampleCount;
		[ReadOnly] public int TraceDepth;
		[ReadOnly] public float3 SkyBottomColor;
		[ReadOnly] public float3 SkyTopColor;
		[ReadOnly] public bool SubPixelJitter;
		[ReadOnly] public ImportanceSampler ImportanceSampler;
		[ReadOnly] public NativeArray<Entity> Entities;
#if BVH
		[ReadOnly] [NativeDisableUnsafePtrRestriction]
		public BvhNode* BvhRoot;
#endif
		[ReadOnly] public PerlinData PerlinData;
#if BVH_ITERATIVE
		[ReadOnly] public int NodeCount;
#endif
		[ReadOnly] public NativeArray<float4> InputColor;
		[ReadOnly] public NativeArray<float3> InputNormal;
		[ReadOnly] public NativeArray<float3> InputAlbedo;

		[WriteOnly] public NativeArray<float4> OutputColor;
		[WriteOnly] public NativeArray<float3> OutputNormal;
		[WriteOnly] public NativeArray<float3> OutputAlbedo;

		[WriteOnly] public NativeArray<Diagnostics> OutputDiagnostics;

#if PATH_DEBUGGING
		[ReadOnly] public int2 DebugCoordinates;
		[NativeDisableUnsafePtrRestriction] public DebugPath* DebugPaths;
#endif

		public void Execute(int index)
		{
			if (CancellationToken[0])
				return;

			// ReSharper disable once PossibleLossOfFraction
			float2 coordinates = float2(
				index % Size.x, // column
				index / Size.x // row
			);

			// early out
			if ((int) coordinates.y % SliceDivider != SliceOffset)
				return;

			float4 lastColor = InputColor[index];
			float3 colorAcc = lastColor.xyz;
			float3 normalAcc = InputNormal[index];
			float3 albedoAcc = InputAlbedo[index];

			int sampleCount = (int) lastColor.w;

			// big primes stolen from Unity's random class
			var rng = new Random((Seed * 0x8C4CA03Fu) ^ (uint) (index * 0x7383ED49u));

#if BVH_ITERATIVE
			BvhNode** nodes = stackalloc BvhNode*[NodeCount];
			Entity* entities = stackalloc Entity[Entities.Length];

			var workingArea = new WorkingArea
			{
				Nodes = nodes,
				Entities = entities,
			};
#endif

#if PATH_DEBUGGING
			int2 intCoordinates = (int2) coordinates;
			bool doDebugPaths = all(intCoordinates == DebugCoordinates);
			if (doDebugPaths)
				for (int i = 0; i < TraceDepth; i++) DebugPaths[i] = default;
#endif

			float3* emissionStack = stackalloc float3[TraceDepth];
			float3* attenuationStack = stackalloc float3[TraceDepth];

			float3 fallbackAlbedo = default, fallbackNormal = default;
			Diagnostics diagnostics = default;

			for (int s = 0; s < SampleCount; s++)
			{
				float2 normalizedCoordinates = (coordinates + (SubPixelJitter ? rng.NextFloat2() : 0.5f)) / Size;
				Ray eyeRay = Camera.GetRay(normalizedCoordinates, ref rng);

				if (Sample(eyeRay, ref rng, emissionStack, attenuationStack,
#if BVH_ITERATIVE
					workingArea,
#endif
#if PATH_DEBUGGING
					doDebugPaths && s == 0, s,
#endif
					out float3 sampleColor, out float3 sampleNormal, out float3 sampleAlbedo, ref diagnostics))
				{
					colorAcc += sampleColor;
					normalAcc += sampleNormal;
					albedoAcc += sampleAlbedo;

					sampleCount++;
				}

				if (s == 0)
				{
					fallbackNormal = sampleNormal;
					fallbackAlbedo = sampleAlbedo;
				}
			}

			OutputColor[index] = float4(colorAcc, sampleCount);
			OutputNormal[index] = sampleCount == 0 ? fallbackNormal : normalAcc;
			OutputAlbedo[index] = sampleCount == 0 ? fallbackAlbedo : albedoAcc;

			OutputDiagnostics[index] = diagnostics;
		}

		bool Sample(Ray eyeRay, ref Random rng, float3* emissionStack, float3* attenuationStack,
#if BVH_ITERATIVE
			WorkingArea workingArea,
#endif
#if PATH_DEBUGGING
			bool doDebugPaths, int sampleIndex,
#endif
			out float3 sampleColor, out float3 sampleNormal, out float3 sampleAlbedo, ref Diagnostics diagnostics)
		{
			float3* emissionCursor = emissionStack;
			float3* attenuationCursor = attenuationStack;

			int depth = 0;
			int? explicitSamplingTarget = null;

			bool firstNonSpecularHit = false;
			sampleNormal = sampleAlbedo = default;

			Ray ray = eyeRay;

			for (; depth < TraceDepth; depth++)
			{
#if BVH
				bool hit = BvhRoot->Hit(Entities,
#else
				bool hit = Entities.Hit(
#endif
					ray, 0, float.PositiveInfinity, ref rng,
#if BVH_ITERATIVE
					workingArea,
#endif
#if FULL_DIAGNOSTICS && BVH_ITERATIVE
					ref diagnostics,
#endif
					out HitRecord rec);

				diagnostics.RayCount++;

				if (hit)
				{
#if PATH_DEBUGGING
					if (doDebugPaths)
						DebugPaths[depth] = new DebugPath { From = ray.Origin, To = rec.Point };
#endif
					// we explicitely sampled an entity and could not hit it -- early out of this sample
					if (explicitSamplingTarget.HasValue && explicitSamplingTarget != rec.EntityId)
						break;

					Material material = Entities[rec.EntityId].Material;
					float3 emission = material.Emit(rec.Point, rec.Normal, PerlinData);
					*emissionCursor++ = emission;

					bool didScatter = material.Scatter(ray, rec, ref rng, PerlinData,
						out float3 albedo, out Ray scatteredRay);

					if (!firstNonSpecularHit)
					{
						if (material.IsPerfectSpecular)
						{
							// TODO: fresnel for dielectric, first diffuse bounce for metallic
						}
						else
						{
							sampleAlbedo = material.Type == MaterialType.DiffuseLight ? emission : albedo;
							sampleNormal = rec.Normal;
							firstNonSpecularHit = true;
						}
					}

					if (!didScatter)
					{
						attenuationCursor++;
						break;
					}

					if (ImportanceSampler.Mode == ImportanceSamplingMode.None || material.IsPerfectSpecular)
					{
						*attenuationCursor++ = albedo;
						ray = scatteredRay;
					}
					else
					{
						float3 outgoingLightDirection = -ray.Direction;
						float scatterPdfValue =
							material.Pdf(scatteredRay.Direction, outgoingLightDirection, rec.Normal);

						ImportanceSampler.Sample(scatteredRay, outgoingLightDirection, rec, material, ref rng,
							out ray, out float pdfValue, out explicitSamplingTarget);

						// scatter ray is likely parallel to the surface, and division would cause a NaN
						if (pdfValue.AlmostEquals(0))
						{
							attenuationCursor++;
							break;
						}

						*attenuationCursor++ = albedo * scatterPdfValue / pdfValue;
					}

					ray = ray.OffsetTowards(dot(scatteredRay.Direction, rec.Normal) >= 0 ? rec.Normal : -rec.Normal);
				}
				else
				{
#if PATH_DEBUGGING
					if (doDebugPaths)
						DebugPaths[depth] = new DebugPath { From = ray.Origin, To = ray.Direction * 99999 };
#endif
					// sample the sky color
					float3 hitSkyColor = lerp(SkyBottomColor, SkyTopColor, 0.5f * (ray.Direction.y + 1));
					*emissionCursor++ = hitSkyColor;
					attenuationCursor++;

					if (!firstNonSpecularHit)
					{
						sampleAlbedo = hitSkyColor;
						sampleNormal = -ray.Direction;
					}

					break;
				}
			}

			sampleColor = 0;

			// safety : if we don't hit an emissive surface within the trace depth limit, fail this sample
			if (depth == TraceDepth)
				return false;

			// attenuate colors from the tail of the hit stack to the head
			while (emissionCursor != emissionStack)
			{
				sampleColor *= *--attenuationCursor;
				sampleColor += *--emissionCursor;
			}

			return true;
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct CombineJob : IJobParallelFor
	{
		static readonly float3 NoSamplesColor = new float3(1, 0, 1);
		static readonly float3 NaNColor = new float3(0, 1, 1);

		public bool DebugMode;
		public bool LdrMode;

		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float4> InputColor;
		[ReadOnly] public NativeArray<float3> InputNormal;
		[ReadOnly] public NativeArray<float3> InputAlbedo;
		[ReadOnly] public int2 Size;

		[WriteOnly] public NativeArray<float3> OutputColor;
		[WriteOnly] public NativeArray<float3> OutputNormal;
		[WriteOnly] public NativeArray<float3> OutputAlbedo;

		public void Execute(int index)
		{
			if (CancellationToken[0])
				return;

			float4 inputColor = InputColor[index];
			var realSampleCount = (int) inputColor.w;

			float3 finalColor;
			if (!DebugMode)
			{
				if (realSampleCount == 0) finalColor = 0;
				else if (any(isnan(inputColor))) finalColor = 0;
				else finalColor = inputColor.xyz / realSampleCount;
			}
			else
			{
				if (realSampleCount == 0)
				{
					int tentativeIndex = index;

					// look-around (for interlaced buffer)
					while (realSampleCount == 0 && (tentativeIndex -= Size.x) >= 0)
					{
						inputColor = InputColor[tentativeIndex];
						realSampleCount = (int) inputColor.w;
					}
				}

				if (realSampleCount == 0) finalColor = NoSamplesColor;
				else if (any(isnan(inputColor))) finalColor = NaNColor;
				else finalColor = inputColor.xyz / realSampleCount;
			}

			float3 finalAlbedo = InputAlbedo[index] / max(realSampleCount, 1);

			if (LdrMode)
			{
				finalColor = min(finalColor, 1);
				finalAlbedo = min(finalAlbedo, 1);
			}

			OutputColor[index] = finalColor;
			OutputNormal[index] = normalizesafe(InputNormal[index] / max(realSampleCount, 1));
			OutputAlbedo[index] = finalAlbedo;
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct ClearBufferJob<T> : IJob where T : unmanaged
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[WriteOnly] public NativeArray<T> Buffer;

		public void Execute()
		{
			if (CancellationToken[0])
				return;

			Buffer.ZeroMemory();
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct CopyFloat3BufferJob : IJob
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> Input;
		[WriteOnly] public NativeArray<float3> Output;

		public void Execute()
		{
			if (CancellationToken[0])
				return;

			NativeArray<float3>.Copy(Input, Output);
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct CopyFloat4BufferJob : IJob
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float4> Input;
		[WriteOnly] public NativeArray<float4> Output;

		public void Execute()
		{
			if (CancellationToken[0])
				return;

			NativeArray<float4>.Copy(Input, Output);
		}
	}

	// because the OIDN API uses strings, we can't use Burst here
	struct OpenImageDenoiseJob : IJob
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> InputColor, InputNormal, InputAlbedo;
		[ReadOnly] public ulong Width, Height;

		public NativeArray<float3> OutputColor;

		public OidnFilter DenoiseFilter;

		public unsafe void Execute()
		{
			if (CancellationToken[0])
				return;

			OidnFilter.SetSharedImage(DenoiseFilter, "color", new IntPtr(InputColor.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);
			OidnFilter.SetSharedImage(DenoiseFilter, "normal", new IntPtr(InputNormal.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);
			OidnFilter.SetSharedImage(DenoiseFilter, "albedo", new IntPtr(InputAlbedo.GetUnsafeReadOnlyPtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);

			OidnFilter.SetSharedImage(DenoiseFilter, "output", new IntPtr(OutputColor.GetUnsafePtr()),
				OidnBuffer.Format.Float3, Width, Height, 0, 0, 0);

			OidnFilter.Commit(DenoiseFilter);
			OidnFilter.Execute(DenoiseFilter);
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct OptixDenoiseJob : IJob
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> InputColor, InputAlbedo;
		[ReadOnly] public uint2 BufferSize;
		[ReadOnly] public OptixDenoiserSizes DenoiserSizes;

		[WriteOnly] public NativeArray<float3> OutputColor;

		public OptixDenoiser Denoiser;
		public CudaStream CudaStream;

		public CudaBuffer InputColorBuffer,
			InputAlbedoBuffer,
			OutputColorBuffer,
			ScratchMemory,
			DenoiserState;

		[BurstDiscard]
		static void Check(CudaError cudaError)
		{
			if (cudaError != CudaError.Success)
				Debug.LogError($"CUDA Error : {cudaError}");
		}

		public unsafe void Execute()
		{
			if (CancellationToken[0])
				return;

			Check(CudaBuffer.Copy(new IntPtr(InputColor.GetUnsafeReadOnlyPtr()), InputColorBuffer.Handle,
				InputColor.Length * sizeof(float3), CudaMemcpyKind.HostToDevice));
			Check(CudaBuffer.Copy(new IntPtr(InputAlbedo.GetUnsafeReadOnlyPtr()), InputAlbedoBuffer.Handle,
				InputAlbedo.Length * sizeof(float3), CudaMemcpyKind.HostToDevice));

			var colorImage = new OptixImage2D
			{
				Data = InputColorBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};
			var albedoImage = new OptixImage2D
			{
				Data = InputAlbedoBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};

			OptixImage2D* optixImages = stackalloc OptixImage2D[2];
			optixImages[0] = colorImage;
			optixImages[1] = albedoImage;

			OptixDenoiserParams denoiserParams = default;

			var outputImage = new OptixImage2D
			{
				Data = OutputColorBuffer,
				Format = OptixPixelFormat.Float3,
				Width = BufferSize.x, Height = BufferSize.y,
				RowStrideInBytes = (uint) (sizeof(float3) * BufferSize.x),
				PixelStrideInBytes = (uint) sizeof(float3)
			};

			OptixDenoiser.Invoke(Denoiser, CudaStream, &denoiserParams, DenoiserState, DenoiserSizes.StateSizeInBytes,
				optixImages, 2, 0, 0,
				&outputImage, ScratchMemory, DenoiserSizes.RecommendedScratchSizeInBytes);

			Check(CudaBuffer.Copy(OutputColorBuffer.Handle, new IntPtr(OutputColor.GetUnsafePtr()),
				OutputColor.Length * sizeof(float3), CudaMemcpyKind.DeviceToHost));
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct FinalizeTexturesJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<bool> CancellationToken;

		[ReadOnly] public NativeArray<float3> InputColor;
		[ReadOnly] public NativeArray<float3> InputNormal;
		[ReadOnly] public NativeArray<float3> InputAlbedo;

		[WriteOnly] public NativeArray<RGBA32> OutputColor;
		[WriteOnly] public NativeArray<RGBA32> OutputNormal;
		[WriteOnly] public NativeArray<RGBA32> OutputAlbedo;

		public void Execute(int index)
		{
			if (CancellationToken[0])
				return;

			// TODO: tone-mapping
			float3 outputColor = saturate(InputColor[index].LinearToGamma()) * 255;
			OutputColor[index] = new RGBA32
			{
				r = (byte) outputColor.x,
				g = (byte) outputColor.y,
				b = (byte) outputColor.z,
				a = 255
			};

			float3 outputNormal = saturate((InputNormal[index] * 0.5f + 0.5f).LinearToGamma()) * 255;
			OutputNormal[index] = new RGBA32
			{
				r = (byte) outputNormal.x,
				g = (byte) outputNormal.y,
				b = (byte) outputNormal.z,
				a = 255
			};

			float3 outputAlbedo = saturate(InputAlbedo[index].LinearToGamma()) * 255;
			OutputAlbedo[index] = new RGBA32
			{
				r = (byte) outputAlbedo.x,
				g = (byte) outputAlbedo.y,
				b = (byte) outputAlbedo.z,
				a = 255
			};
		}
	}

	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast)]
	struct ReduceRayCountJob : IJob
	{
		[ReadOnly] public NativeArray<Diagnostics> Diagnostics;
		[WriteOnly] public NativeArray<int> TotalRayCount;

		public void Execute()
		{
			float totalRayCount = 0;

			for (int i = 0; i < Diagnostics.Length; i++)
				totalRayCount += Diagnostics[i].RayCount;

			TotalRayCount[0] = (int) totalRayCount;
		}
	}

	struct RecordTimeJob : IJob
	{
		[DllImport("Kernel32.dll", CallingConvention = CallingConvention.Winapi)]
		static extern void GetSystemTimePreciseAsFileTime(out long filetime);

		[ReadOnly] public int Index;

		[WriteOnly] public NativeArray<long> Buffer;

		public void Execute()
		{
			GetSystemTimePreciseAsFileTime(out long fileTime);
			Buffer[Index] = fileTime;
		}
	}
}