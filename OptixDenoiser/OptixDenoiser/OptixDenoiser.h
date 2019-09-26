#ifdef OPTIXDENOISER_EXPORTS
#define OPTIXDENOISER_API __declspec(dllexport)
#else
#define OPTIXDENOISER_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" 
{
#endif

OPTIXDENOISER_API void __cdecl fullTest(OptixLogCallback logCallback);

// Creates an Optix device context
OPTIXDENOISER_API OptixDeviceContext __cdecl createDeviceContext(OptixLogCallback logCallback, int logLevel);

// Destroys an Optix device context
OPTIXDENOISER_API OptixResult __cdecl destroyDeviceContext(OptixDeviceContext context);

// Creates a CUDA Stream
OPTIXDENOISER_API cudaError_t __cdecl createCudaStream(cudaStream_t* stream);

// Destroys a CUDA Stream
OPTIXDENOISER_API cudaError_t __cdecl destroyCudaStream(cudaStream_t stream);

// Creates a denoiser object with the given options.
OPTIXDENOISER_API OptixResult __cdecl createDenoiser(OptixDeviceContext context, const OptixDenoiserOptions* options, OptixDenoiser* denoiser);

// Destroys the denoiser object and any associated host resources.
OPTIXDENOISER_API OptixResult __cdecl destroyDenoiser(OptixDenoiser denoiser);

// Sets the model of the denoiser.
// If the kind is OPTIX_DENOISER_MODEL_KIND_USER, then the data and sizeInByes must not be nulland zero respectively.For other kinds, these parameters must be zero.
OPTIXDENOISER_API OptixResult __cdecl setDenoiserModel(OptixDenoiser denoiser, OptixDenoiserModelKind kind, void* data, size_t sizeInBytes);

OPTIXDENOISER_API OptixResult __cdecl computeIntensity(OptixDenoiser denoiser, CUstream stream, const OptixImage2D* inputImage, CUdeviceptr outputIntensity, 
	CUdeviceptr scratch, size_t scratchSizeInBytes);

// Invokes denoiser on a set of input dataand produces one output image.Scratch memory must be available during the execution of the denoiser.
OPTIXDENOISER_API OptixResult __cdecl invokeDenoiser(
	OptixDenoiser denoiser, CUstream stream, const OptixDenoiserParams* params, CUdeviceptr denoiserState, size_t denoiserStateSizeInBytes,
	const OptixImage2D* inputLayers, unsigned int numInputLayers, unsigned int inputOffsetX, unsigned int inputOffsetY, const OptixImage2D* outputLayer,
	CUdeviceptr scratch, size_t scratchSizeInBytes);

// Computes the GPU memory resources required to execute the denoiser.
OPTIXDENOISER_API OptixResult __cdecl computeMemoryResources(OptixDenoiser denoiser, unsigned int outputWidth, unsigned int outputHeight,
	OptixDenoiserSizes* returnSizes);

#ifdef __cplusplus
}
#endif