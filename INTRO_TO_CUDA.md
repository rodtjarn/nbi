# CUDA Architecture Overview
## Streaming Multiprocessors, Warps, and Threads

This document introduces the core concepts behind **CUDA GPU architecture**.
Understanding how GPUs execute programs is essential for writing efficient CUDA code.

CUDA was developed by NVIDIA to allow programmers to use GPUs for **general-purpose computing**, not just graphics.

---

# 1. CPU vs GPU Architecture

CPUs and GPUs are designed for different workloads.

| CPU | GPU |
|----|----|
| Few powerful cores | Thousands of lightweight cores |
| Optimized for low latency | Optimized for high throughput |
| Complex control logic | Massive parallel execution |
| Large caches | Many arithmetic units |

CPUs are optimized for sequential tasks, while GPUs are optimized for **parallel workloads**.

---

# 2. CUDA Programming Model

A CUDA program consists of two parts:

| Component | Runs On |
|-----------|---------|
| Host code | CPU |
| Device code | GPU |

The CPU launches work on the GPU using a **kernel**.

Example:

```cpp
kernel<<<numBlocks, threadsPerBlock>>>(args);
````

This launches thousands of threads on the GPU.

---

# 3. GPU High-Level Architecture

A CUDA-capable GPU consists of multiple **Streaming Multiprocessors (SMs)**.

```
GPU
 ├── SM
 │    ├── CUDA cores
 │    ├── warp schedulers
 │    ├── registers
 │    ├── shared memory
 │
 ├── L2 cache
 └── global memory (VRAM)
```

Each SM executes many threads in parallel.

---

# 4. Streaming Multiprocessors (SM)

The **SM** is the primary execution unit of the GPU.

Each SM contains:

* CUDA cores for arithmetic operations
* Warp schedulers
* Registers
* Shared memory
* Load/store units

Blocks of threads are assigned to SMs during execution.

---

# 5. Threads

A **thread** is the smallest execution unit.

Each thread:

* Executes the kernel code
* Has its own registers
* Has a unique ID

Example kernel:

```cpp
__global__ void add(int *a, int *b, int *c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
```

If 256 threads run this kernel, 256 additions happen in parallel.

---

# 6. Thread Blocks

Threads are grouped into **blocks**.

Execution hierarchy:

```
Kernel
 └ Grid
     └ Block
         └ Thread
```

Example launch:

```cpp
kernel<<<32, 256>>>();
```

Total threads:

```
32 blocks × 256 threads = 8192 threads
```

Threads within a block can:

* synchronize
* share memory

---

# 7. Warps

Inside the GPU hardware, threads execute in groups called **warps**.

```
warp = 32 threads
```

Threads in a warp execute instructions together.

Example block:

```
256 threads per block
256 / 32 = 8 warps
```

Warps are the unit scheduled by the hardware.

---

# 8. SIMT Execution Model

CUDA uses **SIMT (Single Instruction Multiple Threads)**.

Threads execute the same instruction simultaneously across a warp.

Example:

```
warp
thread0 → execute instruction
thread1 → execute instruction
thread2 → execute instruction
...
thread31 → execute instruction
```

This allows GPUs to execute large numbers of threads efficiently.

---

# 9. Warp Divergence

Threads in a warp normally follow the same execution path.
When threads follow different branches, **warp divergence** occurs.

Example:

```cpp
if (threadIdx.x % 2)
    do_A();
else
    do_B();
```

This condition sends:

* **odd threads → do_A()**
* **even threads → do_B()**

Within one warp:

```
thread 0  -> B
thread 1  -> A
thread 2  -> B
thread 3  -> A
...
thread 30 -> B
thread 31 -> A
```

Because different threads in the warp take different branches, the GPU executes both paths sequentially:

1. Execute `do_A()` with only the odd threads active
2. Execute `do_B()` with only the even threads active

Threads that are not participating in the current branch are temporarily disabled.

This reduces efficiency because only part of the warp is active at a time.

Another example:

```cpp
if (threadIdx.x < 16)
    do_A();
else
    do_B();
```

This divides the warp into two groups:

```
threads 0..15  -> A
threads 16..31 -> B
```

Warp divergence still occurs because different threads follow different control paths.

When all threads follow the same branch, divergence does not occur:

```cpp
if (blockIdx.x % 2)
    do_A();
else
    do_B();
```

All threads in the block see the same `blockIdx.x`, so every thread in the warp executes the same instructions.

Key points:

* A warp contains **32 threads**
* Warps execute instructions **in lockstep**
* Divergence occurs when threads follow **different branches**
* Divergent branches execute **sequentially**

---

# 10. Memory Hierarchy

CUDA has multiple memory types with different speeds.

| Memory        | Speed     | Scope         |
| ------------- | --------- | ------------- |
| Registers     | Fastest   | Per thread    |
| Shared memory | Very fast | Per block     |
| L1 cache      | Fast      | Per SM        |
| L2 cache      | Medium    | Entire GPU    |
| Global memory | Slow      | Entire device |

Approximate latency:

```
registers    ~1 cycle
shared mem   ~20 cycles
L2 cache     ~200 cycles
global mem   ~400–800 cycles
```

Efficient CUDA programs minimize accesses to global memory.

---

# 11. Kernel Execution Flow

The typical CUDA execution flow:

```
CPU launches kernel
      ↓
Grid created
      ↓
Blocks assigned to SMs
      ↓
Blocks divided into warps
      ↓
Warp schedulers execute instructions
```

The GPU dynamically schedules warps to keep the hardware busy.

---

# 12. Occupancy

**Occupancy** measures how many warps are active on an SM.

```
occupancy = active warps / maximum warps
```

Higher occupancy helps hide memory latency.

Occupancy depends on:

* register usage
* shared memory usage
* block size

---

# 13. Example CUDA Program

Minimal CUDA program:

```cpp
#include <stdio.h>

__global__ void hello()
{
    printf("Hello from thread %d\n", threadIdx.x);
}

int main()
{
    hello<<<1,8>>>();
    cudaDeviceSynchronize();
}
```

Output:

```
Hello from thread 0
Hello from thread 1
Hello from thread 2
...
Hello from thread 7
```

Each thread runs the same kernel code independently.

---

# Summary

Key CUDA architecture concepts:

* GPUs contain many **Streaming Multiprocessors**
* Each SM executes many **warps**
* A warp contains **32 threads**
* Threads are organized into **blocks and grids**
* Divergent control flow causes **warp divergence**
* Efficient programs minimize divergence and global memory access

```

---

If you'd like, I can also produce a **second version of this `.md` optimized for slides** (so it works perfectly with your Neovim `MarkdownPreview` as a presentation).
```

