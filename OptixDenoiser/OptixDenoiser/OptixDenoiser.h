#ifdef OPTIXDENOISER_EXPORTS
#define OPTIXDENOISER_API __declspec(dllexport)
#else
#define OPTIXDENOISER_API __declspec(dllimport)
#endif

#ifdef __cplusplus    
extern "C" 
{
#endif

// Creates an Optix device context
OPTIXDENOISER_API OptixDeviceContext __cdecl createContext(OptixLogCallback logCallback, int logLevel);

// Destroys an Optix device context
OPTIXDENOISER_API OptixResult __cdecl destroyContext(OptixDeviceContext context);

// Creates a denoiser object with the given options.
OPTIXDENOISER_API OptixResult __cdecl createDenoiser(OptixDeviceContext context, const OptixDenoiserOptions* options, OptixDenoiser* denoiser);

// Destroys the denoiser object and any associated host resources.
OPTIXDENOISER_API OptixResult __cdecl destroyDenoiser(OptixDenoiser denoiser);

#ifdef __cplusplus    
}
#endif