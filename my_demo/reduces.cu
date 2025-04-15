#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

void checkCudaError(const char *msg) {
  cudaError_t __err = cudaGetLastError();
  if (__err != cudaSuccess) {
    fprintf(stderr, "Fatal Error: %s (%s at %s:%d)", msg,
            cudaGetErrorString(__err), __FILE__, __LINE__);
    exit(1);
  }
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_fp32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp32(float *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  // define shared memory
  __shared__ float reduce_smem[NUM_WARPS];
  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // calculate warp level reduction
  sum = warp_reduce_sum_fp32<WARP_SIZE>(sum);
  // store it in the warp leader
  if (lane == 0) {
    reduce_smem[warp] = sum;
  }
  __syncthreads();
  // calculate final answer in the first warp
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_fp32<WARP_SIZE>(sum);
  // Now we have the Sum in this block, then we should add it to the answer: y.
  if (tid == 0)
    atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_fp32x4(float *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  // reduce in every warp.
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
  sum = warp_reduce_sum_fp32<WARP_SIZE>(sum);
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();
  // final sum in first warp
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_fp32(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

int main() {
  const int N = 1024;
  float *h_a = (float *)malloc(N * sizeof(float));
  float *h_y = (float *)malloc(sizeof(float));
  float *d_a, *d_y;

  // 初始化数据
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f; // 简单测试数据
  }
  *h_y = 0.0f;

  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_y, sizeof(float));
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);

  // 调用内核
  const int NUM_THREADS = 256/4;
  const int numBlocks = (N + NUM_THREADS - 1) / NUM_THREADS;
  block_all_reduce_sum_fp32x4<<<numBlocks, NUM_THREADS>>>(d_a, d_y, N);
  checkCudaError("Kernel execution failed");

  // 拷回结果
  cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Sum: %f\n", *h_y);

  // 清理
  cudaFree(d_a);
  cudaFree(d_y);
  free(h_a);
  free(h_y);

  return 0;
}