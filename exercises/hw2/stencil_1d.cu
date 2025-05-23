#include <stdio.h>
#include <algorithm>

using namespace std;

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int *in, int *out) {
    __shared__ int temp[BLOCK_SIZE+2*RADIUS];
    // 因为我们把地址传进来的时候已经加了RADIUS，这里gindex就不用加了。
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
      temp[lindex - RADIUS] = in[gindex - RADIUS];
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[lindex + offset];

    // Store the result
    // 因为我们的d_in , d_out只偏移了RADIUS，所以最后RADIUS个取不到，就不会做多余操作。
    out[gindex] = result;
}

void fill_ints(int *x, int n) {
  fill_n(x, n, 1);
}

int main(void) {
  int *in, *out; // host copies of a, b, c
  int *d_in, *d_out; // device copies of a, b, c

  // Alloc space for host copies and setup values
  int size = (N + 2*RADIUS) * sizeof(int);
  in = (int *)malloc(size); fill_ints(in, N + 2*RADIUS);
  out = (int *)malloc(size); fill_ints(out, N + 2*RADIUS);

  // Alloc space for device copies
  cudaMalloc((void **)&d_in, size);
  cudaMalloc((void **)&d_out, size);

  // Copy to device
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

  // Launch stencil_1d() kernel on GPU
  // 注意这里也要加上偏移量RADIUS  不然两头就会偏大
  stencil_1d<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in+RADIUS, d_out+RADIUS);

  // Copy result back to host
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  // Error Checking
  for (int i = 0; i < N + 2*RADIUS; i++) {
    if (i<RADIUS || i>=N+RADIUS){
      if (out[i] != 1)
    	printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
    } else {
      if (out[i] != 1 + 2*RADIUS)
    	printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2*RADIUS);
    }
  }

  // Cleanup
  free(in); free(out);
  cudaFree(d_in); cudaFree(d_out);
  printf("Success!\n");
  return 0;
}
