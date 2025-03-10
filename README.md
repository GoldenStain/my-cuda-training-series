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