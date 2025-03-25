#include <stdio.h>

// 复用之前的warp_reduce_sum
template <int WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float tmp = __shfl_up_sync(0xffffffff, val, offset);
        float old_val = val;
        if (threadIdx.x >= offset) val += tmp;
        printf("T%d: offset=%d, tmp=%.1f, val=%.1f->%.1f\n", threadIdx.x, offset,
               tmp, old_val, val);
    }
    return val;
}

__global__ void warp_reduce_demo() {
    const int idx = threadIdx.x;
    float val = idx + 1;  // 初始值1~32

    printf("-- Before reduce: T%d val=%.1f\n", idx, val);
    float sum = warp_reduce_sum<32>(val);
}

int main() {
    warp_reduce_demo<<<1, 32>>>();  // 单个Warp
    cudaDeviceSynchronize();
    return 0;
}
