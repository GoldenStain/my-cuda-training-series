#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cfloat>
#include <climits>
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

// DataStructure required for online-softmax
struct alignas(8) MD {
  float m, d;
};
// 公式链接：https://zhuanlan.zhihu.com/p/638788074

// Reduce for online softmax
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  const unsigned int mask = 0xffffffff;
#pragma unroll
  for (int stride = mask >> 1; stride; stride >>= 1) {
    MD other;
    other.d = __shfl_xor_sync(mask, value.d, stride);
    other.m = __shfl_xor_sync(mask, value.m, stride);

    bool value_bigger = (value.m > other.m);
    MD bigger_md = value_bigger ? value : other;
    MD smaller_md = value_bigger ? other : value;

    value.d = bigger_md.d + smaller_md.d * __expf(smaller_md.m - bigger_md.m);
    value.m = bigger_md.m;
  }
  return value;
}

// Block Reduce Sum
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_fp32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int tid = threadIdx.x;
  int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  float value = warp_reduce_sum_fp32<WARP_SIZE>(val);

  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_fp32(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// Block Reduce Max
template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_fp32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int tid = threadIdx.x;
  int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  float value = warp_reduce_max_fp32(val);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_max_fp32(value);
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// naive softmax
template <const int NUM_THREADS = 256>
__global__ void softmax_fp32_per_token_kernel(float *x, float *y, int N) {
  int tid = threadIdx.x, idx = blockIdx.x * NUM_THREADS + tid;

  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum_fp32(exp_val);
  if (idx < N)
    y[idx] = exp_val / exp_sum;
}

// float4 naive softmax
template <const int NUM_THREADS = 256 / 4>
__global__ void softmax_fp32x4_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x, idx = (blockIdx.x * NUM_THREADS + tid) << 2;
  float4 reg_x = FLOAT4(x), reg_exp;
  reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;
  float value = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
  float exp_sum = block_reduce_sum_fp32<NUM_THREADS>(value);
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;
    reg_y.z = reg_exp.z / exp_sum;
    reg_y.w = reg_exp.w / exp_sum;
    FLOAT4(y[idx]) = reg_y;
  }
}

// safe-softmax
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_fp32_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x, idx = blockIdx.x * NUM_THREADS + tid;
  float val = (idx < N) ? x[idx] : -FLT_MAX;
  float max_val = block_reduce_max_fp32(val);
  float exp_val = (idx < N) ? expf(x[idx] - max_val) : 0.0F;
  float exp_sum = block_reduce_sum_fp32<NUM_THREADS>(exp_val);
  if (idx < N)
    y[idx] = exp_val / exp_sum;
}

// float4 version of safe-softmax
template <const int NUM_THREADS = 256 / 4>
__global__ void safe_softmax_fp32x4_per_token_kernel(float *x, float *y,
                                                     int N) {
  const int tid = threadIdx.x, idx = (blockIdx.x * NUM_THREADS + tid) << 2;
  float4 reg_x = FLOAT4(x[idx]);
  reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
  reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
  reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
  reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;
  float val = reg_x.x;
  val = fmaxf(val, reg_x.y);
  val = fmaxf(val, reg_x.z);
  val = fmaxf(val, reg_x.w);
  float max_val = block_reduce_max_fp32<NUM_THREADS>(val);
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z - max_val) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w - max_val) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_fp32(exp_val);
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;
    reg_y.z = reg_exp.z / exp_sum;
    reg_y.w = reg_exp.w / exp_sum;
    FLOAT4(y[idx]) = reg_y;
  }
}

// online-softmax
template <const int NUM_THREADS = 256>
__global__ void online_safe_softmax_f32_per_token_kernel(float *x, float *y,
                                                         int N) {
  const int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
  const int WARP_NUM = NUM_THREADS / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE, warp_id = tid / WARP_SIZE;
  MD val;
  val.m = (tid < N) ? x[idx] : -FLT_MAX;
  val.d = (tid < N) ? 1.0f : 0.0f;

  __shared__ MD shared[WARP_NUM];
  MD res = warp_reduce_md_op<WARP_SIZE>(val);

  if (lane_id == 0) {
    shared[warp_id] = res;
  }
  __syncthreads();

  if (tid < WARP_SIZE) {
    MD block_res = shared[lane_id];
    block_res = warp_reduce_md_op<WARP_NUM>(block_res);
    if (lane_id == 0)
        shared[0] = block_res;
  }
  __syncthreads();

  MD final_res = shared[0];
  float denominator = __fdividef(1.0f, final_res.d);
  if (idx < N) {
    y[idx] = __expf(x[idx] - final_res.m) * denominator;
  }
}

// TODO: add float4 version of online-softmax

// CPU参考实现（按照你的kernel逻辑）
void softmax_cpu(const float *x, float *y, int N) {
  float max_val = -INFINITY;
  float sum = 0.0f;

  // 计算max（尽管你的kernel未实现）
  for (int i = 0; i < N; ++i) {
    max_val = fmaxf(max_val, x[i]);
  }

  // 按照你的kernel实际逻辑：直接计算exp后求和（未减max）
  for (int i = 0; i < N; ++i) {
    sum += expf(x[i]);
  }

  for (int i = 0; i < N; ++i) {
    y[i] = expf(x[i]) / sum;
  }
}
// 数值比较（容忍FP32精度误差）
bool allclose(const float *a, const float *b, int N, float rtol = 1e-4,
              float atol = 1e-5) {
  for (int i = 0; i < N; ++i) {
    if (fabs(a[i] - b[i]) > (atol + rtol * fabs(b[i]))) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                << std::endl;
      return false;
    }
  }
  return true;
}
int main() {
  constexpr int N = 256; // 测试数据长度（必须<= NUM_THREADS）
  constexpr int NUM_THREADS = 256;

  // 生成测试数据（注意数值范围，避免exp溢出）
  float *host_x = new float[N];
  float *host_y_gpu = new float[N];
  float *host_y_cpu = new float[N];

  // 测试用例1：全零输入（简单验证）
  for (int i = 0; i < N; ++i)
    host_x[i] = 0.0f;

  // 测试用例2：递增序列（验证数值稳定性）
  // for (int i = 0; i < N; ++i) host_x[i] = i * 0.1f;

  // 分配GPU内存
  float *device_x, *device_y;
  CUDA_CHECK(cudaMalloc(&device_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_y, N * sizeof(float)));

  // CPU参考计算
  softmax_cpu(host_x, host_y_cpu, N);

  // GPU计算
  CUDA_CHECK(
      cudaMemcpy(device_x, host_x, N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 grid((N + NUM_THREADS - 1) / NUM_THREADS);
  dim3 block(NUM_THREADS);
  softmax_fp32_per_token_kernel<NUM_THREADS>
      <<<grid, block>>>(device_x, device_y, N);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError()); // 捕获kernel错误

  CUDA_CHECK(cudaMemcpy(host_y_gpu, device_y, N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 结果验证
  bool pass = allclose(host_y_gpu, host_y_cpu, N);
  std::cout << "Test " << (pass ? "PASSED" : "FAILED") << std::endl;

  // 边界测试：空指针（应崩溃）
  // softmax_fp32_per_token_kernel<NUM_THREADS><<<1, 1>>>(nullptr, nullptr, 0);

  // 清理资源
  delete[] host_x;
  delete[] host_y_gpu;
  delete[] host_y_cpu;
  CUDA_CHECK(cudaFree(device_x));
  CUDA_CHECK(cudaFree(device_y));

  return 0;
}
