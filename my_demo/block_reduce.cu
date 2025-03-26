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

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int WARP_SIZE = 32;
    __shared__ float shared[NUM_WARPS];
    
    // Warp内规约
    val = warp_reduce_sum<WARP_SIZE>(val);
    if (threadIdx.x % WARP_SIZE == 0) {
        printf("Warp %d sum=%.1f\n", threadIdx.x/WARP_SIZE, val);
        shared[threadIdx.x/WARP_SIZE] = val;
    }
    __syncthreads();

    // 跨Warp规约
    val = (threadIdx.x < NUM_WARPS) ? shared[threadIdx.x] : 0;
    printf("Post shared: T%d val=%.1f\n", threadIdx.x, val);
    val = warp_reduce_sum<32>(val); // 强制使用32线程
    
    return val;
}

__global__ void block_reduce_demo() {
    const int idx = threadIdx.x;
    float val = idx + 1; // 初始值1~128
    
    float sum = block_reduce_sum<4>(val); // 128/32=4 warps
    
    // if (idx == 0) {
    //     printf("\n*** Final Block Sum = %.1f ***\n", sum);
    // }
    printf("\n***On thread%d, the sum is : %.1f***\n", idx, sum);
}

int main() {
    block_reduce_demo<<<1, 128>>>(); // 128线程=4 warps
    cudaDeviceSynchronize();
    return 0;
}
