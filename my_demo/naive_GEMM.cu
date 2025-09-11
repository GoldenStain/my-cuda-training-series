// 定义naive gemm的kernel函数
__global__ void naiveSgemm(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, const int M, const int N,
                           const int K) {

  // 当前thread在C矩阵中的row
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  // 当前thread在C矩阵中的col
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M && n < N) {
    float psum = 0.0;
// 告知编译器自动展开循环体，这样可以减少循环控制的开销（循环次数小的时候可以这么做）
#pragma unroll
    // 取出A[row]和B[col]，然后逐个元素相乘累加，得到最终结果
    for (int k = 0; k < K; k++) {
      // a[OFFSET(m, k, K)]: 获取A[m][k]
      // b[OFFSET(k, n, N)]: 获取B[k][n]
      psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = psum;
  }
}

const int BM = 32, BN = 32;
const int M = 512, N = 512, K = 512;
dim3 blockDim(BN, BM);
dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);