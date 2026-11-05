# Lecture 10 · LLM Serving Basics II：Intro to GPUs

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/10_llm_serving_basics_2a.pdf`](../slides/10_llm_serving_basics_2a.pdf)
> 目标是建立直觉，不深入细节。

## 本讲要点

- GPU 是 **throughput machine**：放弃单线程速度，换取成千上万个简单核心
- 每个 GPU 受三项规格限制：**peak compute、HBM bandwidth、memory capacity**
- **Roofline model**：ridge point（H200 约 206 FLOP/B）区分 memory-bound 与 compute-bound
- **Decode 是 bandwidth-bound**：batching 能提升 weight GEMV 的 arithmetic intensity，但 decode attention 仍是 memory-bound
- 趋势：新 GPU 的 ridge point 右移，越来越多 kernel 变成 bandwidth-bound
- H200 架构：SM、warp、Tensor Core、memory hierarchy、Little's law、TMA；多 GPU 的 scale-up 与 scale-out

---

## 1. GPU Basics

### CPU vs GPU

- CPU 为**单线程**优化（大量 control、out-of-order、branch prediction、大 cache）—— 像跑车，载 2 人最快到达
- GPU 为**整个芯片的吞吐**优化（大部分面积给算术单元）—— 像公交车队，慢但能运几千人

| | 64 核服务器 CPU | NVIDIA H200 SXM |
|---|---|---|
| 并行单元 | 64 cores | 132 SMs，16,896 FP32 cores |
| 同时在运行的线程 | — | 270,336（每 SM 2,048） |
| 内存带宽 | ~0.46 TB/s（DDR5） | 4.8 TB/s（HBM3e） |
| 峰值 dense 算力 | ~5 TFLOPs FP32 | 989 TFLOPs BF16 |
| 隐藏延迟的方式 | cache、out-of-order | **切换到就绪的 warp** |

**为什么适合 LLM**：每个 token 都在同样的权重上并行执行同样的 matmul。

### Warp 切换隐藏内存延迟

- 一个 warp 等待内存时，另一个 warp 发射指令，计算单元保持忙碌
- A100/H200/B200：每个 SM 有 4 个 warp scheduler，每个最多管理 16 个 warps → 每 SM 64 个 resident warps

### SIMT（Single Instruction, Multiple Threads）

- 一个 warp = 32 个线程；标量代码只写一次（`y[i] = a*x[i] + b`）
- 一次 fetch、一次 decode、一次调度决策驱动 32 个 lanes —— 一条 warp 指令完成 32 条标量指令的工作，类似 CPU 的 SIMD

### Branch Divergence

- `if (i % 4 == 0) f() else g()`：warp 会**依次执行两条路径**，f 只用 8/32 个 lanes，g 用 24/32 个，其余被 mask 掉
- 分支发散会损失吞吐

### GPU 的取舍

| GPU 放弃 | 换来 |
|---|---|
| 单线程速度（时钟一般，无 speculation，串行代码慢很多） | 面积和功耗给了成千上万个算术单元 |
| 并行必须显式（约 27 万线程才能填满 132 SMs，小任务会让芯片空闲） | 通过切换 warp 隐藏延迟，而不是靠 cache 和 speculation |
| 控制流共享（32 个线程一组，分支发散损失吞吐） | 一次指令 fetch/decode 驱动 32 个 cores |
| 每线程片上内存小（满 occupancy 时约 240 B/线程） | 每个线程的状态常驻，warp 切换零开销 |

### 三项决定性规格

| | A100 | H200 | B200 | Rubin |
|---|---|---|---|---|
| Peak BF16 compute | 312 TFLOP/s | 989 | 2,250 | ≈ 4,000 |
| Memory bandwidth | 2.0 TB/s | 4.8 | 8 | ≈ 22 |
| Memory capacity | 80 GB HBM2e | 141 GB HBM3e | 180 GB HBM3e | 288 GB HBM4 |

直觉：**compute 是引擎，bandwidth 是油管，capacity 是油箱**。

### Compute 只有在每个字节喂给足够多 FLOPs 时才能达到峰值

以 8B 模型（16 GB weights）为例：
- **Decode, batch 1**：每 token 3.3 ms；读 16 GB 权重，Tensor Cores **99.5% 时间空闲**（0.016 ms 的计算在等 3.3 ms 的读取）—— 每个权重只用一次
- **Prefill, 2,048 tokens**：每个 prompt 33 ms，33 TFLOP，接近峰值 —— 每个权重喂给 2,048 个 tokens，读取很快结束，计算决定速度

**Kernel time ≈ max(bytes ÷ bandwidth, FLOPs ÷ peak)**

### Roofline Model

- **Arithmetic intensity** = FLOPs ÷ 与 HBM 之间搬运的字节数
- **Ridge point** = peak FLOP/s ÷ bandwidth（H200 ≈ 989 / 4.8 ≈ **206 FLOP/B**）
- Ridge 左侧：**memory-bound**，速度 = bandwidth × intensity
- Ridge 右侧：**compute-bound**，速度 = peak FLOP/s
- Decode 在左侧，prefill 在右侧（Williams et al., Roofline, CACM 2009）

### 常见 Arithmetic Intensity（每元素 b 字节）

| 运算 | Intensity |
|---|---|
| Vector add，长度 N | N ÷ 3Nb = 1/(3b) |
| GEMV（decode），N × N，每个权重用一次 | 2N² ÷ (N² + 2N)b ≈ 2/b |
| Decode attention，n × d，每个 KV 每个 head 用一次 | 4nd ÷ 2ndb = 2/b |
| Square GEMM，N × N，每个元素复用约 N 次 | 2N³ ÷ 3N²b = 2N/(3b) |

### Batching 对 decode 有帮助吗？

- Batching 能把 **weight GEMV** 的 intensity 提升到接近 ridge（batch 256 时约 256 FLOP/B）
- 但 **decode attention 始终是 2/b**：每个请求有自己的 KV cache，无法共享

### Context length 的影响

Llama 3 8B、batch 16、BF16、H200：
- Weights 固定 16.1 GB，每步读一次
- KV cache 每 token 128 KiB，随 batch × L 增长
- 每步 HBM 读取：1K → 18.2 GB（3.8 ms）；32K → 84.8 GB（17.7 ms）

### 小测：H200 上 batch-1 decode 用了多少峰值算力？

答案：**< 1%（约 0.5%）**。Batch-1 GEMV 约 1 FLOP/B，只能达到约 5 / 989 TFLOP/s。
提升方法：**提高 arithmetic intensity** —— 更大的 batch、quantization、减少 KV 流量。大 GEMM 已在平顶上，只能靠更快的计算或更少的 FLOPs。

### 趋势：低精度把 ridge 右移

- B200 BF16 ridge ≈ 281；FP4 下 ridge ≈ 1,125；MI355X ≈ 1,259；Rubin dense NVFP4 需要约 **1,600 FLOP/B** 才能 compute-bound
- 从 A100 到 Rubin，BF16 算力增长约 13 倍，带宽只增长约 9 倍 → **kernel 越来越 bandwidth-bound**

### 小结

1. GPU 是 throughput machine
2. 三项上限：peak compute、HBM bandwidth、memory capacity
3. Roofline：ridge point 区分 memory-bound 与 compute-bound
4. Decode：batching 让 weight 读取接近 ridge，KV-cache 读取仍 memory-bound
5. 趋势：新 GPU ridge 右移

---

## 2. GPU Architecture（以 H200 为例）

### 整体

- 132 个 SMs，每个有自己的 registers 和 256 KB SRAM
- 共享一个 **50 MB L2 cache**
- **HBM3e**：6 stacks × 24 GB，可用 141 GB，4.8 TB/s
- **NVLink**：到其他 GPU 900 GB/s

### SM 的四个 partition

- 每个 SM 分成 4 个 partition，每个 partition 有一个 warp scheduler、64 KB registers、32 个 FP32 cores、1 个 **Tensor Core**；每周期各发射一条 warp 指令
- 共享 256 KB L1 / shared memory
- 最多 64 warps（2,048 threads）常驻，scheduler 每周期挑一个就绪 warp

### Tensor Cores

- 一条 warp 指令完成 `D = A × B + C`（如 16×16 × 16×8）—— 一次 `mma` = 2,048 次乘加，是一次 FMA 的 64 倍
- **Matmul 占 LLM FLOPs 的 99% 以上**；softmax 和 norm 用 FP32 cores
- 为什么快：
  - 每 FLOP 指令更少（摊薄 decode、调度、操作数处理）
  - 专用乘加数据通路，并行处理整个矩阵 tile
  - 高效的操作数复用（A 的每个值喂多列输出，B 的每个值喂多行）
  - 低精度运算（FP16、BF16、FP8 提高吞吐、缩小操作数，通常用更高精度累加）

### Memory Hierarchy（H200 / Hopper）

| 层级 | 容量 | 延迟 | 带宽 |
|---|---|---|---|
| Registers | — | 在流水线中 | 每 core 每周期 3 个操作数 |
| Shared memory / L1 | 每 SM 256 KB | ≈ 30 cycles（L1 hit 40） | 128 B/clk/SM，132 SMs 约 33 TB/s |
| L2 cache | 50 MB | ≈ 250 cycles | ≈ 12 TB/s |
| HBM3e | 141 GB | ≈ 500 cycles | 4.8 TB/s |
| Host 内存与其他 GPU | — | 微秒级 | NVLink 4 每方向 450 GB/s；PCIe 5.0 ≈ 64 GB/s |

### Little's Law：内存是一根 250 ns 的管道

**bandwidth = bytes in flight ÷ latency**

- Naive kernel：每 SM 4 KB in flight（32 warps × 32 threads × 一次 4B load）→ 2.2 TB/s，峰值的 45%
- 满管道：每 SM 需要 9 KB in flight（4.8 TB/s × 250 ns ÷ 132）→ 4.8 TB/s
- 硬件固定了管道长度，**提升带宽的唯一方法是让更多字节在管道中**
- 方法：16 字节 vector loads；每线程发四个独立 load；bulk asynchronous copy（**TMA**）

每代需要更多 bytes in flight：H200 每 SM 9 KB（GPU 共 1.2 MB）；B200 14 KB（2.0 MB）；Rubin 25 KB（5.5 MB）。

### Advanced：Hopper / Blackwell 的数据通路特性

| 特性 | 作用 | 效果 |
|---|---|---|
| Coalescing | 32 个线程读连续 4 字节合并成一次 128 B 请求 | 更少、更满的内存请求 |
| 16-byte vector loads | 一条 warp 指令搬 512 B 而不是 128 B | 每指令 4 倍字节 |
| cp.async（Ampere+） | 全局内存直接拷到 shared memory，不经 registers | registers 空出来做更多 load |
| **TMA**（Hopper） | 一个线程发起整个 tile 的 bulk copy，硬件生成地址 | 一条指令几十 KB in flight |
| TMA multicast | 一次 L2 读取落到 cluster 中每个 SM 的 shared memory（最多 8 个 thread blocks） | 每 SM 的 L2 流量减少 |
| **TMEM**（Blackwell） | 结果存放在 256 KB tensor memory 而不是 registers | registers 空出来保持更多 load in flight |
| **2-SM MMA**（Blackwell） | 两个 SM 共同完成一个乘法，各取一半 tile | tile 加倍 |

### 从单 GPU 到多 GPU：Scale-up 与 Scale-out

| | 8-GPU node（HGX H200） | NVL72 rack（GB200） |
|---|---|---|
| 结构 | 8 GPU 通过 NVSwitch 互联；8 个 400 Gb/s NIC 连到 scale-out 网络 | 72 GPU 通过 NVLink switch 互联（每排 4 GPU + 2 CPU） |
| HBM | 4.8 TB/s | 8 TB/s |
| NVLink | 900 GB/s | 1.8 TB/s |
| RDMA | 50 GB/s | 100 GB/s |

### Advanced：NVIDIA GPU 二十年

- 2006–17（图形 → 通用计算）：Tesla（CUDA、thread blocks、shared memory）→ Fermi（L1/L2、ECC）→ Kepler（warp shuffle）→ Maxwell（SM 分四个 partition）→ Pascal（HBM2、NVLink）→ Volta（**Tensor Cores**）
- 2018–26（计算 → 低精度 AI 引擎）：Turing（INT8/INT4）→ Ampere（TF32、BF16、2:4 sparsity、cp.async、MIG）→ Hopper（FP8、TMA、clusters、wgmma）→ Blackwell（双 die、FP6/FP4、TMEM、NVL72）→ Blackwell Ultra（288 GB、15 PF NVFP4）→ Rubin（HBM4、NVLink 6）
- G80 → B300：峰值算力增长约 43,000 倍，带宽只增长约 90 倍 —— **算力远远跑赢内存**

---

## 3. Optional：其他加速器

| 芯片 | 特点 | 编程方式 |
|---|---|---|
| **AMD Instinct MI355X** | 8 个 compute die 封装为一个 GPU；256 MB Infinity Cache；64 线程 wavefront；5.0 PF dense FP8、288 GB、8 TB/s | ROCm、HIP（类 CUDA，hipify 转换）；PyTorch、vLLM、SGLang 可运行；Triton 或 Composable Kernel |
| **Google TPU Ironwood** | **Systolic array**（weight-stationary：权重留在单元中，activations 从左进入，partial sums 向下流动）；3D torus 互联最多 9,216 芯片；4.6 PF FP8、192 GB、7.4 TB/s | JAX、PyTorch/XLA；Pallas 写自定义 kernel |
| **Amazon Trainium3** | 8 个 NeuronCores，各有 Tensor（systolic）、Vector、Scalar 引擎；每核 32 MB SRAM 由 DMA 填充，**没有硬件 cache**；2.5 PF MXFP8 | Neuron SDK；NKI（Python kernel，显式在 HBM 和 SRAM 间搬 tile）；没有 CUDA |
| **Cerebras WSE-3** | 整片晶圆一颗芯片，44 GB on-chip SRAM，无片外内存 | 大模型需要流式加载权重 |
| **Groq LPU** | 230 MB SRAM，无片外内存 | 8-bit 70B 模型需约 300 颗芯片 |
| **Microsoft Maia 200** | 272 MB SRAM + 216 GB HBM3e | 仅在 Azure 中可用 |
