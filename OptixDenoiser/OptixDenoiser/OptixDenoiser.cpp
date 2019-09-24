#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "OptixDenoiser.h"

OPTIXDENOISER_API OptixDeviceContext createContext(OptixLogCallback logCallbackFunction, int logLevel)
{
	OptixDeviceContext context = nullptr;
	{
		cudaError_t error;
		if ((error = cudaFree(0)) != cudaSuccess)
		{
			if (logCallbackFunction != nullptr)
				logCallbackFunction(1, "CUDA Error", cudaGetErrorString(error), nullptr);
			return nullptr;
		}

		CUcontext cuCtx = 0;  // zero means take the current context
		OptixResult result = optixInit();
		if (result != OPTIX_SUCCESS)
		{
			if (logCallbackFunction != nullptr)
				logCallbackFunction(1, optixGetErrorName(result), optixGetErrorString(result), nullptr);
			return nullptr;
		}
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = logCallbackFunction;
		options.logCallbackLevel = 4;
		result = optixDeviceContextCreate(cuCtx, &options, &context);
		if (result != OPTIX_SUCCESS)
		{
			if (logCallbackFunction != nullptr)
				logCallbackFunction(1, optixGetErrorName(result), optixGetErrorString(result), nullptr);
			return nullptr;
		}
	}
	return context;
}

OPTIXDENOISER_API OptixResult destroyContext(OptixDeviceContext context)
{
	return optixDeviceContextDestroy(context);
}

OPTIXDENOISER_API OptixResult createDenoiser(OptixDeviceContext context, const OptixDenoiserOptions* options, OptixDenoiser* denoiser)
{
	return optixDenoiserCreate(context, options, denoiser);
}

OPTIXDENOISER_API OptixResult destroyDenoiser(OptixDenoiser denoiser)
{
	return optixDenoiserDestroy(denoiser);
}
