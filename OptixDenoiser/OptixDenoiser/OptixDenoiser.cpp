#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "OptixDenoiser.h"

OPTIXDENOISER_API cudaError_t initializeCuda()
{
	return cudaFree(0);
}

OPTIXDENOISER_API cudaError_t resetCudaDevice()
{
	return cudaDeviceReset();
}

OPTIXDENOISER_API OptixResult initializeOptix()
{
	return optixInit();
}

OPTIXDENOISER_API OptixResult createDeviceContext(OptixDeviceContextOptions options, OptixDeviceContext* context)
{
	CUcontext cuCtx = 0;  // zero means take the current context
	return optixDeviceContextCreate(cuCtx, &options, context);
}

OPTIXDENOISER_API OptixResult destroyDeviceContext(OptixDeviceContext context)
{
	return optixDeviceContextDestroy(context);
}

OPTIXDENOISER_API cudaError_t createCudaStream(cudaStream_t* stream)
{
	return cudaStreamCreate(stream);
}

OPTIXDENOISER_API cudaError_t destroyCudaStream(cudaStream_t stream)
{
	return cudaStreamDestroy(stream);
}

OPTIXDENOISER_API OptixResult createDenoiser(OptixDeviceContext context, const OptixDenoiserOptions* options, OptixDenoiser* denoiser)
{
	return optixDenoiserCreate(context, options, denoiser);
}

OPTIXDENOISER_API OptixResult destroyDenoiser(OptixDenoiser denoiser)
{
	return optixDenoiserDestroy(denoiser);
}

OPTIXDENOISER_API OptixResult setDenoiserModel(OptixDenoiser denoiser, OptixDenoiserModelKind kind, void* data, size_t sizeInBytes)
{
	return optixDenoiserSetModel(denoiser, kind, data, sizeInBytes);
}

OPTIXDENOISER_API OptixResult computeIntensity(OptixDenoiser denoiser, CUstream stream, const OptixImage2D* inputImage, CUdeviceptr outputIntensity, 
	CUdeviceptr scratch, size_t scratchSizeInBytes)
{
	return optixDenoiserComputeIntensity(denoiser, stream, inputImage, outputIntensity, scratch, scratchSizeInBytes);
}

OPTIXDENOISER_API OptixResult setupDenoiser(OptixDenoiser denoiser, CUstream stream, unsigned int outputWidth, unsigned int outputHeight, 
	CUdeviceptr denoiserState, size_t denoiserStateSizeInBytes, CUdeviceptr scratch, size_t scratchSizeInBytes)
{
	return optixDenoiserSetup(denoiser, stream, outputWidth, outputHeight, denoiserState, denoiserStateSizeInBytes, scratch, scratchSizeInBytes);
}

OPTIXDENOISER_API OptixResult invokeDenoiser(
	OptixDenoiser denoiser, CUstream stream, const OptixDenoiserParams* params, CUdeviceptr denoiserState, size_t denoiserStateSizeInBytes,
	const OptixImage2D* inputLayers, unsigned int numInputLayers, unsigned int inputOffsetX, unsigned int inputOffsetY, const OptixImage2D* outputLayer, 
	CUdeviceptr scratch, size_t scratchSizeInBytes)
{
	return optixDenoiserInvoke(denoiser, stream, params, denoiserState, denoiserStateSizeInBytes, inputLayers, numInputLayers, inputOffsetX, inputOffsetY,
		outputLayer, scratch, scratchSizeInBytes);
}

OPTIXDENOISER_API OptixResult computeMemoryResources(OptixDenoiser denoiser, unsigned int outputWidth, unsigned int outputHeight, 
	OptixDenoiserSizes* returnSizes)
{
	return optixDenoiserComputeMemoryResources(denoiser, outputWidth, outputHeight, returnSizes);
}

OPTIXDENOISER_API cudaError_t allocateCudaBuffer(size_t sizeInBytes, CUdeviceptr* outPointer)
{
	return cudaMalloc(reinterpret_cast<void**>(outPointer), sizeInBytes);
}

OPTIXDENOISER_API cudaError_t copyCudaBuffer(const void* source, void* destination, size_t size, cudaMemcpyKind kind)
{
	return cudaMemcpy(destination, source, size, kind);
}

OPTIXDENOISER_API cudaError_t deallocateCudaBuffer(CUdeviceptr buffer)
{
	return cudaFree(reinterpret_cast<void*>(buffer));
}