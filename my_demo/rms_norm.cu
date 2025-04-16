#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val))[0])

void checkCudaError(const char *msg) {
  cudaError_t __err = cudaGetLastError();
  if (__err != cudaSuccess) {
    fprintf(stderr, "Fatal Error: %s (%s at %s:%d)", msg,
            cudaGetErrorString(__err), __FILE__, __LINE__);
    exit(1);
  }
}

// 错误检查宏
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d code=%d (%s)\n", __FILE__,          \
              __LINE__, err, cudaGetErrorString(err));                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int WARP_SIZE = 32;

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_fp32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_fp32(float val) {
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int tid = threadIdx.x, idx = blockIdx.x * NUM_THREADS + tid;
  const int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
  static __shared__ float smem[NUM_WARPS];
  // warp_reduce
  val = warp_reduce_sum_fp32<WARP_SIZE>(val);
  if (lane == 0)
    smem[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
  val = warp_reduce_sum_fp32<NUM_THREADS>(val);
  val = __shfl_sync(0xffffffff, val, 0, 32);
  return val;
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_fp32_kernel(float *x, float *y, float g, int N,
                                     int K) {
  const int tid = threadIdx.x, bid = blockIdx.x, idx = blockDim.x * bid + tid;
  const float eps = 1e-5f;

  __shared__ float s_variance;
  float value = (idx < N * K) ? x[idx] : 0.0f;
  float variance = value * value;
  variance = block_reduce_sum_fp32(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + eps);
  // wait for shared memory to be ready.
  __syncthreads();
  if (idx < N * K)
    y[idx] = g * (value * s_variance);
}

template <const int NUM_THREADS = 256 / 4>
__global__ void rms_norm_fp32x4_kernel(float *x, float *y, float g, int N,
                                       int K) {
  const int bid = blockIdx.x, tid = threadIdx.x,
            idx = (bid * blockDim.x + tid) << 2;
  const float eps = 1e-5f;

  __shared__ float s_variance;
  float4 reg_x = FLOAT4(x[idx]);
  float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y +
                                    reg_x.z * reg_x.z + reg_x.w * reg_x.w)
                                 : 0.0f;
  variance = block_reduce_sum_fp32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + eps);
  __syncthreads();
  float4 reg_y;
  reg_y.x = reg_x.x * s_variance * g;
  reg_y.y = reg_y.y * s_variance * g;
  reg_y.z = reg_y.z * s_variance * g;
  reg_y.w = reg_y.w * s_variance * g;
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}