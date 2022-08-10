#ifndef EM_CUDA_RNG
#define EM_CUDA_RNG

__device__ void xorshift32(unsigned long &state){
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
}

__device__ void xorshift64(unsigned long long &state){
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
}

__device__ double random_double(unsigned long long &state) {
    // Returns a random real in [0,1).
    xorshift64(state);
    return state / ((double )0xffffffffffffffffULL + 1.0);
}

__device__ double random_double(double min, double max, unsigned long long &state) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double(state);
}

#ifndef EM_RNG_COMMONS
#define EM_RNG_COMMONS


inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}
#endif
#endif