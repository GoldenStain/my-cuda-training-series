# CUDA Training Resource
The materials in this repository accompany the CUDA Training Series presented at ORNL and NERSC.

You can find the slides and presentation recordings at https://www.olcf.ornl.gov/cuda-training-series/

## 小笔记

## lecture 1 & 2

Idx.x和Idx.y分别表示x方向和y方向，所以我们应该用Idx.y来表示行，Idx.x来表示列。

### lecture 3

sp unit(cores)计算单精度，ld unit计算双精度。

CPU的概念映射到GPU中，并不是cores，GPU的core更类似于cpu当中的ALU，功能甚至比alu更少

gpu中和CPU的core相对应的概念是SM(streaming multiprocessor).

每个SM能持有的warp数量有限（硬件）-> 每个block能持有的thread数量有限（软件）

通过起足够数量的线程，来hide latency.

### lecture 4

L1缓存一个line是128bytes，而一个warp有32个线程，128/4=32.

CUDA GPU的一个segment大小为32bytes.

shared memory的组织形式：
32 banks, 4-byte wide banks.
其上限通常是48KB

如果大量的线程都要访问同一个bank的资源，导致其因为串行调度而变慢，我们可以使用名叫"padding"的技巧，来打乱bank的内存布局，从而使我们的访问重新并行化。

## lecture 5

`Transformation`: 输出和输入规模相同，我们只需要一个线程负责一个输出就好了

`reduction`：输出规模远小于输入规模，我们需要考虑新的并行策略。