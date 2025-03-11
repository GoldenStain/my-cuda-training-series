#include <stdio.h>
#include <chrono>
#include <iostream>
#include <iomanip>

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

const int DSIZE = 32 * 1048576;
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds) {
    // 注意这里的步长是gridDim.x * blockDim.x ，这是我们第一次遇到gridDim.x
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < ds;
         idx += gridDim.x * blockDim.x) {  // a grid-stride loop
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    using namespace std::chrono;
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE];  // allocate space for vectors in host memory
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];
    for (int i = 0; i < DSIZE; i++) {  // initialize vectors in host memory
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }
    cudaMalloc(&d_A,
               DSIZE * sizeof(float));  // allocate device space for vector A
    cudaMalloc(&d_B,
               DSIZE * sizeof(float));  // allocate device space for vector B
    cudaMalloc(&d_C,
               DSIZE * sizeof(float));  // allocate device space for vector C
    cudaCheckErrors("cudaMalloc failure");  // error checking
    // copy vector A to device:
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    // copy vector B to device:
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    // cuda processing sequence step 1 is complete
    int threads = 512;  // modify this line for experimentation
    int blocks = 16;     // modify this line for experimentation
    auto start = high_resolution_clock::now();
    vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);
    auto end = high_resolution_clock::now();
    duration<double, std::milli> elapsed = end - start;
    std::cout << "It took " << std::fixed << std::setprecision(3) << elapsed.count() << "milliseconds" << std::endl;
    cudaCheckErrors("kernel launch failure");
    // cuda processing sequence step 2 is complete
    //  copy vector C from device to host:
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);
    return 0;
}
