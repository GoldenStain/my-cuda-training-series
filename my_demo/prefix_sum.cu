// naive version
__global__ void ScanAndWritePartSumKernel(const int32_t *input, int32_t *part,
                                          int32_t *output, size_t n,
                                          size_t part_num) {
  for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
    // this part process input[part_begin:part_end]
    // store sum to part[part_i], output[part_begin:part_end]
    size_t part_begin = part_i * blockDim.x;
    size_t part_end = min((part_i + 1) * blockDim.x, n);
    if (threadIdx.x == 0) { // naive implemention
      int32_t acc = 0;
      for (size_t i = part_begin; i < part_end; ++i) {
        acc += input[i];
        // 先让output[i]变成局部前缀和
        output[i] = acc;
      }
      part[part_i] = acc;
    }
  }
}
__global__ void ScanPartSumKernel(int32_t *part, size_t part_num) {
  int32_t acc = 0;
  for (size_t i = 0; i < part_num; ++i) {
    acc += part[i];
    part[i] = acc;
  }
}
__global__ void AddBaseSumKernel(int32_t *part, int32_t *output, size_t n,
                                 size_t part_num) {
  for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
    if (part_i == 0) {
      continue;
    }
    int32_t index = part_i * blockDim.x + threadIdx.x;
    if (index < n) {
      // output[index]已经是当前块内的局部前缀和了，这时候再加上前面所有块的前缀和，就是答案。
      output[index] += part[part_i - 1];
    }
  }
}
// for i in range(n):
//   output[i] = input[0] + input[1] + ... + input[i]
void ScanThenFan(const int32_t *input, int32_t *buffer, int32_t *output,
                 size_t n) {
  size_t part_size = 1024; // tuned
  size_t part_num = (n + part_size - 1) / part_size;
  size_t block_num = std::min<size_t>(part_num, 128);
  // use buffer[0:part_num] to save the metric of part
  int32_t *part = buffer;
  // after following step, part[i] = part_sum[i]
  ScanAndWritePartSumKernel<<<block_num, part_size>>>(input, part, output, n,
                                                      part_num);
  // after following step, part[i] = part_sum[0] + part_sum[1] + ... part_sum[i]
  ScanPartSumKernel<<<1, 1>>>(part, part_num);
  // make final result
  AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num);
}