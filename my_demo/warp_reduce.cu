#include <stdio.h>

// 复用之前的warp_reduce_sum
template <int WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, val, mask);
        printf("T%d: mask=%d, tmp=%.1f, val=%.1f->%.1f\n", 
               threadIdx.x, mask, tmp, val, val + tmp);
        val += tmp;
    }
    return val;
}

__global__ void warp_reduce_demo() {
    const int idx = threadIdx.x;
    float val = idx + 1; // 初始值1~32
    
    printf("-- Before reduce: T%d val=%.1f\n", idx, val);
    float sum = warp_reduce_sum<32>(val);
    
    if (idx == 0) {
        printf("\n** Final Warp Sum = %.1f **\n", sum);
    }
}

int main() {
    warp_reduce_demo<<<1, 32>>>(); // 单个Warp
    cudaDeviceSynchronize();
    return 0;
}
