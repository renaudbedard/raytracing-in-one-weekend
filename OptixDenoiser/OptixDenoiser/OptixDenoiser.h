#ifdef OPTIXDENOISER_EXPORTS
#define OPTIXDENOISER_API __declspec(dllexport)
#else
#define OPTIXDENOISER_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" 
{
#endif
	
// Initializes CUDA functionalities
OPTIXDENOISER_API cudaError_t __cdecl initializeCuda();

// Resets the current CUDA device
OPTIXDENOISER_API cudaError_t __cdecl resetCudaDevice();

// Initializes the OptiX function table
OPTIXDENOISER_API OptixResult __cdecl initializeOptix();

// Creates an Optix device context
OPTIXDENOISER_API OptixResult __cdecl createDeviceContext(OptixDeviceContextOptions options, OptixDeviceContext* context);

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

// Initializes the state required by the denoiser.
OPTIXDENOISER_API OptixResult __cdecl setupDenoiser(OptixDenoiser denoiser, CUstream stream, unsigned int outputWidth, unsigned int outputHeight,
	CUdeviceptr denoiserState, size_t denoiserStateSizeInBytes, CUdeviceptr scratch, size_t scratchSizeInBytes);

// Invokes denoiser on a set of input dataand produces one output image.Scratch memory must be available during the execution of the denoiser.
OPTIXDENOISER_API OptixResult __cdecl invokeDenoiser(
	OptixDenoiser denoiser, CUstream stream, const OptixDenoiserParams* params, CUdeviceptr denoiserState, size_t denoiserStateSizeInBytes,
	const OptixImage2D* inputLayers, unsigned int numInputLayers, unsigned int inputOffsetX, unsigned int inputOffsetY, const OptixImage2D* outputLayer,
	CUdeviceptr scratch, size_t scratchSizeInBytes);

// Computes the GPU memory resources required to execute the denoiser.
OPTIXDENOISER_API OptixResult __cdecl computeMemoryResources(OptixDenoiser denoiser, unsigned int outputWidth, unsigned int outputHeight,
	OptixDenoiserSizes* returnSizes);

// Allocates a device-resident buffer for use with CUDA
OPTIXDENOISER_API cudaError_t __cdecl allocateCudaBuffer(size_t sizeInBytes, CUdeviceptr* outPointer);

// Copies a CUDA buffer from host to device or vice-versa
OPTIXDENOISER_API cudaError_t __cdecl copyCudaBuffer(const void* source, void* destination, size_t size, cudaMemcpyKind kind);

// Deallocates a CUDA buffer
OPTIXDENOISER_API cudaError_t __cdecl deallocateCudaBuffer(CUdeviceptr buffer);

#ifdef __cplusplus
}
#endif