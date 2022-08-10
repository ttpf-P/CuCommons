#ifndef EM_CUDA_COMMONS
#define EM_CUDA_COMMONS

#include "CUrng.hpp"
#include "CUvec3.hpp"

#ifndef EM_UNIVERSAL_CONST
#define EM_UNIVERSAL_CONST
__device__ const double pi = 3.1415926535897932385;
#endif

__device__ double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ double radians_to_degrees(double radians) {
    return radians / pi * 180.0;
}

__device__ double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
             file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#endif