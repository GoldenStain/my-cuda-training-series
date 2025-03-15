#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg)                                        \
    do {                                                            \
        cudaError_t __err = cudaGetLastError();                     \
        if (__err != cudaSuccess) {                                 \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
                    cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n");             \
            exit(1);                                                \
        }                                                           \
    } while (0)

const size_t N = 8ULL * 1024ULL * 1024ULL;  // data size
const int BLOCK_SIZE = 256;                 // CUDA maximum is 1024

__global__ void reduce(float *gdata, float *out, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    size_t now_idx = idx;

    while(now_idx < idx) {
        sdata[tid] = max(sdata[tid], gdata[now_idx]);
        now_idx += gridDim.x * blockDim.x;
    }

    for (int offset = blockDim.x/2; offset; offset>>=1) {
        __syncthreads();
        if (tid < offset)
            sdata[tid] = max(sdata[tid], sdata[tid+offset]);
    }
    
}

int main() {
    float *h_A, *h_sum, *d_A, *d_sums;
    const int blocks = 640;
    h_A = new float[N];  // allocate space for data in host memory
    h_sum = new float;
    float max_val = 5.0f;
    for (size_t i = 0; i < N; i++)  // initialize matrix in host memory
        h_A[i] = 1.0f;
    h_A[100] = max_val;
    cudaMalloc(&d_A, N * sizeof(float));  // allocate device space for A
    cudaMalloc(
        &d_sums,
        blocks * sizeof(float));  // allocate device space for partial sums
    cudaCheckErrors("cudaMalloc failure");  // error checking
    // copy matrix A to device:
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    // cuda processing sequence step 1 is complete
    reduce<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N);
    cudaCheckErrors("reduction kernel launch failure");
    // cuda processing sequence step 2 is complete
    //  copy vector sums from device to host:
    cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda processing sequence step 3 is complete
    cudaCheckErrors(
        "reduction w/atomic kernel execution failure or cudaMemcpy D2H "
        "failure");
    printf(
        "reduction output: %f, expected sum reduction output: %f, expected max "
        "reduction output: %f\n",
        *h_sum, (float)((N - 1) + max_val), max_val);
    return 0;
}
