#include <__clang_cuda_runtime_wrapper.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void checkCudaError(const char *msg) {
  cudaError_t __err = cudaGetLastError();
  if (__err != cudaSuccess) {
    fprintf(stderr, "Fatal Error: %s (%s at %s:%d)", msg,
            cudaGetErrorString(__err), __FILE__, __LINE__);
    exit(1);
  }
}

constexpr int WARP_SIZE = 32;

// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_fp32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_fp32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// Block Reduce Sum
template<const int NUM_THREADS = 256>
__device__ float block

int main() { return 0; }