# Lecture 10 · LLM Serving Basics II：Intro to GPUs

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations（Juncheng Yang，Fall 2026）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/10_llm_serving_basics_2a.pdf`](../slides/10_llm_serving_basics_2a.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 9（prefill / decode、KV cache 大小、Llama 3 的参数量）。本讲的目标是建立直觉，不深入细节。

## 本讲脉络

```
GPU 基础：GPU 是“吞吐机器” → 切换 warp 来隐藏内存延迟 → SIMT 与分支发散 → GPU 做了哪些取舍
→ 三项规格（算力、带宽、容量）→ 什么时候算力能跑满：arithmetic intensity 与 roofline
→ decode 为什么只用到 0.5% 的算力；batching 和 context 长度的影响 → 低精度让 ridge 右移
GPU 架构：H200 全景 → SM 的四个分区 → Tensor Core → 内存层级 → Little's law（在途字节）→ TMA、TMEM 等新特性
→ 从一张卡到很多张卡：scale-up（NVLink）与 scale-out（网络）→ NVIDIA GPU 二十年
→（选读）AMD、Google TPU、Amazon Trainium 与 SRAM 型加速器
```

**一句话总结**：GPU 用**成千上万个简单核心**换掉了 CPU 的少数几个复杂核心，靠**在不同 warp 之间切换**来隐藏内存延迟，擅长大规模并行的矩阵运算。每个 kernel 的速度受**算力**和**显存带宽**两者中较紧的那个限制：**每读一个字节能做多少次运算（arithmetic intensity）** 决定了它落在 roofline 的哪一侧。LLM 的 **decode 是极端的带宽受限**（batch 为 1 时只用到约 0.5% 的算力），所以要 batching、量化、缩小 KV 流量；而新一代 GPU 的算力增长快于带宽，越来越多的 kernel 会变成带宽受限。

---

## 1. GPU Basics

### 1.1 GPU 的形态

| 消费级 | 数据中心级 |
|---|---|
| 一张 PCIe 卡插在台式机里 | 一块板上有 8 张 GPU，装在服务器中 |

### 1.2 GPU 是一台“吞吐机器”

| CPU：为**单个线程**优化 | GPU：为**整颗芯片**的吞吐优化 |
|---|---|
| 大量面积用于控制逻辑、乱序执行（out-of-order）、分支预测 | 大部分面积用于算术单元 |
| 每个核心有大的私有 cache，还有共享的 L3 cache | 每个 SM 有寄存器和小的片上存储，共享一个 L2 cache |
| 内存：几个 DDR 通道 | 内存：HBM 堆栈，数千 bit 宽 |
| 像**跑车**：载 2 个人，最快到达 | 像**公交车队**：慢一些，但一次运几千人 |

**GPU 用少数几个快核心，换来了成千上万个简单核心：**

| | 64 核服务器 CPU | NVIDIA H200 SXM |
|---|---|---|
| 并行单元 | 64 个核心 | 132 个 SM，16,896 个 FP32 核心 |
| 同时在运行的线程 | — | 270,336（每个 SM 2,048 个） |
| 内存带宽 | 约 0.46 TB/s（DDR5） | 4.8 TB/s（HBM3e） |
| 峰值 dense 算力 | 约 5 TFLOPs（FP32） | 989 TFLOPs（BF16） |
| 隐藏延迟的方式 | cache、乱序执行 | **切换到一个就绪的 warp** |

CPU 遇到 cache miss 时，一个线程会继续往前执行后面不依赖这次读取的指令，直到乱序窗口填满才停下；**GPU 则直接切换到另一个就绪的 warp**。

**为什么适合 LLM**：每个 token 都在**同样的权重**上做**同样的矩阵乘法**，天然可以大规模并行。

### 1.3 切换 warp 来隐藏内存延迟

**Warp**：32 个线程组成的一组，是 GPU 调度的基本单位。

```
Warp 0：发射指令 ─ 等内存 ─────────── 发射 ─ 等内存 ───
Warp 1：         发射 ─ 等内存 ─────────── 发射 ─ 等内存
Warp 2：               发射 ─ 等内存 ───────────
Warp 3：                     发射 ─ 等内存 ─────
计算单元：W0 W1 W2 W3 W0 W1 W2 W3 W0 …（始终在忙）
```

**为什么有效**：一个 warp 在等待内存时，另一个 warp 可以发射指令，所以计算单元一直保持忙碌。

A100 / H200 / B200：每个 SM 有 4 个 warp scheduler，每个最多管理 16 个 warp，所以**每个 SM 最多常驻 64 个 warp**。常驻的 warp 越多（**occupancy，占用率**越高），越有机会找到一个就绪的 warp 来填补等待。

### 1.4 SIMT：单指令，多线程

**CUDA 的线程层级**：

```
Grid（一次 kernel 启动的全部线程）
 └─ Thread block（一组线程，最多 1,024 个，运行在同一个 SM 上，共享 shared memory）
     └─ Warp（32 个线程，一起执行同一条指令）
         └─ Thread（每个线程有自己的寄存器）
```

**写一次标量代码，由 32 个线程一起执行**：

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;   // 每个线程算出自己负责的元素下标
y[i] = a * x[i] + b;                              // 每个线程处理一个元素
```

- 一次取指、一次译码、一次调度决策 → 驱动 32 个执行通道（lane 0 到 lane 31），每个 lane 有自己的寄存器
- **为什么高效**：一条 warp 指令完成了 32 条标量指令的工作。控制开销只付一次，由 32 个 lane 分摊，就像 CPU 上的一条 SIMD 向量指令

### 1.5 分支发散（Branch divergence）

```c
int i = threadIdx.x;
bool p = (i % 4 == 0);        // 每 4 个线程中只有 1 个满足
if (p) a = f(x[i]);           // active mask 0x11111111：8 个 lane 执行，24 个闲置
else   a = g(x[i]);           // active mask 0xEEEEEEEE：24 个 lane 执行，8 个闲置
y[i] = a;                     // 重新汇合，32 个 lane 全部执行
```

**代价**：同一个 warp 中的线程走不同的分支时，warp 要**轮流执行两条路径**：f 只用 32 个 lane 中的 8 个，g 用 24 个。两段时间相加，吞吐打了折扣。

**怎么避免**：尽量让同一个 warp 内的线程走同一个分支（例如按条件先把数据分组）。

### 1.6 GPU 背后的取舍

| GPU 放弃了 | 换来了 |
|---|---|
| **单线程速度**：时钟频率一般，没有投机执行，串行代码慢得多 | 面积和功耗都给了成千上万个算术单元 |
| **并行必须是显式的**：约 27 万个线程才能填满 132 个 SM，小任务会让芯片空闲 | 靠切换 warp 隐藏延迟，而不是靠 cache 和投机 |
| **控制流是共享的**：32 个线程一组，分支发散会损失吞吐 | 一次取指和译码驱动 32 个核心 |
| **每个线程的片上存储很小**：满占用时每个线程约 240 字节 | 每个线程的状态都常驻在芯片上，所以 warp 切换**零开销** |

### 1.7 三项规格决定了每张 GPU 的上限

| | A100 | H200 | B200 | Rubin（初步） |
|---|---|---|---|---|
| **峰值 BF16 算力** | 312 TFLOP/s | 989 TFLOP/s | 2,250 TFLOP/s | ≈ 4,000 TFLOP/s |
| **显存带宽** | 2.0 TB/s | 4.8 TB/s | 8 TB/s | ≈ 22 TB/s |
| **显存容量** | 80 GB HBM2e | 141 GB HBM3e | 180 GB HBM3e | 288 GB HBM4 |
| 限制了什么 | 数学运算能多快 | 权重和 KV cache 能多快被读出 | 能放下多少权重和 KV cache | |

**直觉**：**算力是引擎，带宽是油管，容量是油箱。**

### 1.8 只有当读出的每个字节都喂给足够多的 FLOPs，算力才能跑满

以一个 8B 模型（权重 16 GB，BF16）在 H200 上为例：

| | Decode，batch 1 | Prefill，2,048 个 token |
|---|---|---|
| 每次耗时 | **3.3 ms / token** | **33 ms / prompt** |
| HBM 读取 | 16 GB 权重 | 权重读完之后就空闲了 |
| Tensor Core | **99.5% 的时间在空等** | 33 TFLOP，接近峰值 |
| 原因 | 每个权重只用一次，0.016 ms 的计算要等 3.3 ms 的读取 | 每个权重被 2,048 个 token 复用，读取很快完成，计算决定速度 |

计算：
- Decode：读 16 GB ÷ 4.8 TB/s ≈ 3.3 ms；计算 $2 \times 8\text{B} = 16$ GFLOPs ÷ 989 TFLOP/s ≈ 0.016 ms
- Prefill：计算 $2 \times 8\text{B} \times 2048 \approx 33$ TFLOPs ÷ 989 TFLOP/s ≈ 33 ms；读权重仍是 3.3 ms，被计算完全覆盖

$$
\text{Kernel time} \approx \max\Big(\frac{\text{bytes}}{\text{bandwidth}},\; \frac{\text{FLOPs}}{\text{peak}}\Big)
$$

### 1.9 Roofline：判断一个 kernel 受内存还是受算力限制

**Arithmetic intensity（算术强度）** = 执行的 FLOPs ÷ 与 HBM 之间搬运的字节数（单位 FLOP/B）

$$
\text{可达到的 FLOP/s} = \min\big(\text{峰值 FLOP/s},\; \text{带宽} \times \text{算术强度}\big)
$$

```
可达 FLOP/s（对数）
   │          ┌──────────────────── compute roof（峰值 989 TFLOP/s）
   │         ╱  ← prefill 在这里（compute-bound）
   │        ╱
   │       ╱  memory roof（斜率 = 带宽）
   │      ╱
   │  decode 在这里（memory-bound）
   └────────┼──────────────────── 算术强度（对数）
          ridge ≈ 206 FLOP/B（H200）
```

- **Ridge point（屋脊点）** = 峰值 FLOP/s ÷ 带宽。H200：989 ÷ 4.8 ≈ **206 FLOP/B**
- **屋脊点左侧：memory-bound**，速度 = 带宽 × 算术强度
- **屋脊点右侧：compute-bound**，速度 = 峰值算力
- 一个 kernel 想要跑满 H200 的算力，从 HBM 读出的**每个字节要参与约 206 次运算**

（Williams, Waterman & Patterson, “Roofline”, CACM 2009）

### 1.10 常见运算的算术强度

设每个元素占 $b$ 字节（BF16 时 $b = 2$）：

| 运算 | FLOPs ÷ 字节 | 算术强度 | BF16 时 | 结论 |
|---|---|---|---|---|
| **Vector add**（长度 $N$） | $N \div 3Nb$ | $\frac{1}{3b}$ | 约 0.17 | 每搬 3 个元素做 1 次运算，极端 memory-bound |
| **GEMV**（decode，$N \times N$ 矩阵乘向量，每个权重只用一次） | $2N^2 \div (N^2 + 2N)b$ | $\approx \frac{2}{b}$ | 约 1 | memory-bound |
| **Decode attention**（$n \times d$ 的 KV，每个 head 每个 KV 元素只用一次） | $4nd \div 2ndb$ | $\frac{2}{b}$ | 1 | memory-bound |
| **Square GEMM**（$N \times N$，每个元素被复用约 $N$ 次） | $2N^3 \div 3N^2 b$ | $\frac{2N}{3b}$ | $N = 8192$ 时约 2,731 | compute-bound |

### 1.11 Batching 对 decode 有帮助吗？

- **对权重的 GEMV 有帮助**：batch 为 $B$ 时，一次读出的权重被 $B$ 个请求复用，算术强度约为 $B$ FLOP/B（BF16）。batch 256 时达到约 256，**接近屋脊点**
- **对 decode attention 没有帮助**：每个请求有**自己的** KV cache，无法共享，算术强度始终是 $2/b$

所以 batching 让权重读取变得高效，但 KV cache 的读取仍然是 memory-bound —— context 越长、KV 越大，这个问题越突出。

### 1.12 Context 长度对 decode 的影响

Llama 3 8B、batch 16、BF16，每个 decode 步从 HBM 读取的字节数：

| Context 长度 | 权重 | KV cache | 合计 | 每步时间（÷ 4.8 TB/s） |
|---|---|---|---|---|
| 1K | 16.1 GB | 2.1 GB | 18.2 GB | 3.8 ms |
| 4K | 16.1 GB | 8.6 GB | 24.7 GB | — |
| 8K | 16.1 GB | 17.2 GB | 33.3 GB | — |
| 16K | 16.1 GB | 34.4 GB | 50.5 GB | — |
| 32K | 16.1 GB | 68.7 GB | 84.8 GB | 17.7 ms |

- **权重**：固定 16.1 GB，无论 context 多长，每步都读一次
- **KV cache**：每个 token 128 KiB，随 batch × context 长度增长
- 32K 时 KV cache 已经是权重的 4 倍多，**每步时间增加到 4.7 倍**

### 1.13 小测：H200 上 batch 1 的 decode 用了多少算力？

已知：峰值算力 989 TFLOP/s（BF16 dense）；HBM 带宽 4.8 TB/s；batch 1 的 decode 是 GEMV，每读 1 字节权重约做 1 次 FLOP。

**答案：不到 1%（约 0.5%）**。算术强度约 1 FLOP/B，只能达到 $4.8\ \text{TB/s} \times 1 = 4.8$ TFLOP/s，约为 989 的 0.5%。

**怎么提高**：**提高算术强度** —— 更大的 batch、量化（每个权重的字节变少，同样的计算读更少的字节）、减少 KV 流量（GQA、MLA、KV cache 量化）。大的 GEMM 已经顶在平顶上，只能靠更快的数学单元或更少的 FLOPs。

总结：**prefill 是 compute-bound，decode 是 bandwidth-bound。**

### 1.14 趋势：低精度把屋脊点推向右边

屋脊点 = dense 峰值算力 ÷ HBM 带宽（FLOP/B）：

| GPU | HBM 带宽 | FP32 | BF16 | FP4 |
|---|---|---|---|---|
| A100 SXM | 2.04 TB/s | 9.6 | ~153 | — |
| H200 | 4.8 TB/s | — | ~206 | — |
| B200 HGX | 8 TB/s | 9.4 | ~281 | 1,125 |
| MI355X | 8 TB/s | — | — | 1,259 |
| Rubin（初步） | 22 TB/s | — | — | 1,591 |

- 精度越低，峰值算力越高，屋脊点越靠右：Rubin 的 dense NVFP4 需要每字节约 **1,600 次运算**才能 compute-bound
- **算力比带宽长得快**：从 A100 到 Rubin，BF16 算力增长约 13 倍，带宽只增长约 9 倍 → **越来越多的 kernel 会变成带宽受限**

### 1.15 GPU 基础：要点

1. **吞吐机器**：GPU 用快速的单线程换来成千上万个并行核心
2. **三个上限**：峰值算力、HBM 带宽、显存容量限制了每个 kernel
3. **Roofline**：屋脊点（H200 上 206 FLOP/B）区分 memory-bound 和 compute-bound
4. **Decode**：batching 让权重读取接近屋脊点；KV cache 的读取仍然 memory-bound
5. **趋势**：新 GPU 的屋脊点右移，更多 kernel 变成带宽受限

---

## 2. GPU 架构（以 H200 为例）

### 2.1 全景：132 个 SM，共享一个 50 MB 的 L2 cache 和 141 GB 的 HBM3e

```
HBM3e 24GB  HBM3e 24GB  HBM3e 24GB            （6 个 HBM 堆栈，可用 141 GB，4.8 TB/s）
┌──────────────────────────────────────────┐
│ SM SM SM SM SM SM SM SM SM SM SM ...      │  132 个 SM，每个有自己的寄存器和 256 KB SRAM
│        L2 cache · 50 MB · 所有 SM 共享     │
│ SM SM SM SM SM SM SM SM SM SM SM ...      │
└──────────────────────────────────────────┘
HBM3e 24GB  HBM3e 24GB  HBM3e 24GB
NVLink：与其他 GPU 之间 900 GB/s
```

### 2.2 每个 SM 分成四个分区，每个分区每周期发射一条 warp 指令

```
┌ Partition 1 ─────────────┐ ┌ Partition 2 ─────────────┐
│ Warp scheduler           │ │ Warp scheduler           │
│ Registers 64 KB          │ │ Registers 64 KB          │
│ 32 FP32 cores │ Tensor Core│ │ 32 FP32 cores │ Tensor Core│
└──────────────────────────┘ └──────────────────────────┘
┌ Partition 3 ... ┐ ┌ Partition 4 ... ┐
L1 / shared memory · 256 KB（整个 SM 共享）
```

- **Warp = 32 个线程**：一条指令在 32 个线程上同时执行（SIMT）
- **最多 64 个 warp**：2,048 个线程常驻；scheduler 每个周期挑一个就绪的 warp
- **Tensor Core**：每条指令完成一个小的矩阵乘加

### 2.3 LLM 的数学主要是矩阵乘法，Tensor Core 专门加速它

**一条 warp 指令**（`mma`）：$D = A \times B + C$，其中 $A$ 是 16×16，$B$、$C$、$D$ 是 16×8。

- 一次 `mma` = $16 \times 8 \times 16 = 2{,}048$ 次乘加，是一条 FMA（融合乘加）指令工作量的 64 倍
- **矩阵乘法占 LLM 的 FLOPs 的 99% 以上**；softmax 和归一化用 FP32 核心

H200 SXM 的峰值 dense 算力（TFLOP/s）：

| 单元 | 峰值 |
|---|---|
| FP32 标量 | 约 67 |
| TF32 Tensor | 约 495 |
| BF16 Tensor | 约 989 |
| FP8 Tensor | 约 1,979 |

同样的芯片，用 Tensor Core 和低精度，算力差了一个数量级以上。

**为什么 Tensor Core 快**：

| 原因 | 说明 |
|---|---|
| **每个 FLOP 需要的指令更少** | 一条 MMA 指令包含大量乘加，分摊了译码、调度和操作数处理的开销 |
| **专用的算术单元** | 专门的乘加数据通路，并行处理整个矩阵块 |
| **高效的操作数复用** | A 的每个值参与多个输出列的计算，B 的每个值参与多个输出行的计算 |
| **低精度运算** | FP16、BF16、FP8 提高吞吐、缩小操作数，通常配合更高精度的累加 |

### 2.4 内存层级

| 层级 · H200 容量 | 延迟（Hopper） | 带宽 |
|---|---|---|
| **Registers（寄存器）** | 在流水线中 | 每条指令读取，每个核心每周期 3 个操作数 |
| **Shared memory / L1** · 每 SM 256 KB | 约 30 周期（L1 命中约 40） | 每 SM 每周期 128 B，132 个 SM 合计约 33 TB/s |
| **L2 cache** · 50 MB | 约 250 周期 | Hopper 上约 12 TB/s |
| **HBM3e** · 141 GB，6 个堆栈 | 约 500 周期 | 4.8 TB/s |
| **Host 内存与其他 GPU** | 微秒级 | NVLink 4 每方向 450 GB/s；PCIe 5.0 约 64 GB/s |

越往下容量越大、越慢。高性能 kernel（如 FlashAttention）的核心思想就是：**把数据搬到上层（shared memory、寄存器）后尽量多次复用**，减少对 HBM 的访问。

### 2.5 内存是一根 250 ns 长的管道：带宽取决于管道里有多少字节

**Little's law**（排队论的基本定理：系统中的平均数量 = 到达率 × 平均停留时间）：

$$
\text{带宽} = \frac{\text{在途字节数（bytes in flight）}}{\text{延迟}}
$$

一次内存读取从发出到数据到达要约 250 ns（管道的长度由硬件决定）。要跑满带宽，管道里必须**同时装着足够多的字节**：

| 情况 | 每 SM 在途字节 | 带宽 |
|---|---|---|
| **Naive kernel**：32 个 warp × 32 个线程 × 每线程一次 4 B 读取 | 4 KB | 4 KB × 132 ÷ 250 ns ≈ **2.2 TB/s（峰值的 45%）** |
| **满管道**：4.8 TB/s × 250 ns ÷ 132 个 SM | **约 9 KB** | **4.8 TB/s（100%）** |

**硬件固定了管道的长度，所以提高带宽的唯一办法是让管道里装更多的字节。**

**增加在途字节的方法**：
- 使用 **16 字节的向量读取**（一次读 16 B 而不是 4 B）
- 每个线程发出**四个相互独立的读取**（不用等上一个返回再发下一个）
- 使用**批量异步拷贝（TMA）**

### 2.6 每一代都需要更多的在途字节才能跑满带宽

| | H200 · Hopper | B200 · Blackwell | Rubin |
|---|---|---|---|
| HBM | 141 GB HBM3e | 180 GB HBM3e | 288 GB HBM4 |
| HBM 带宽 | 4.8 TB/s | 8 TB/s | 22 TB/s |
| 整个 GPU 的在途字节 | 1.2 MB | 2.0 MB | 5.5 MB |
| 每个 SM 的在途字节 | 9 KB | 14 KB | 25 KB |
| 新的数据通路特性 | TMA、clusters | TMEM、2-SM MMA | 内联 TMA 描述符 |

### 2.7 进阶：Hopper 与 Blackwell 如何增加每条指令的在途字节

| 特性 | 作用 | 效果 |
|---|---|---|
| **Coalescing（合并访问）** | 32 个线程读取连续的 4 字节字，合并成一次 128 B 的请求 | 更少、更满的内存请求 |
| **16-byte vector loads** | 一条 warp 指令搬 512 B 而不是 128 B | 每条指令 4 倍字节 |
| **cp.async**（Ampere 起） | 把全局内存直接拷进 shared memory，不经过寄存器 | 寄存器空出来，可以发更多读取 |
| **TMA**（Hopper） | 一个线程发起一整块（tile）的批量拷贝，地址由硬件生成 | 一条指令就有几十 KB 在途 |
| **TMA multicast** | 一次 L2 读取，落到一个 cluster 中每个 SM 的 shared memory | 每个 SM 的 L2 流量减少 |

- **TMA**：例如一个 128 × 64 的 tile，一个线程发起，硬件自动完成整块拷贝；寄存器保持空闲
- **Cluster multicast**：最多 8 个 thread block 组成一个 cluster，共享一次 L2 读取（SM 0–3 都拿到同一块数据），L2 流量下降

**Blackwell 的两个新特性**：
- **TMEM（Tensor Memory）**：Hopper 中矩阵乘的结果放在寄存器里；Blackwell 把结果放进一个 256 KB 的专用 tensor memory，**寄存器被释放出来，可以维持更多在途的读取**
- **2-SM MMA**：两个 SM 合作完成一次矩阵乘，各自加载共享 tile 的一半，相当于 tile 大小翻倍、复用率更高

### 2.8 从一张卡到很多张卡：scale-up 与 scale-out

| | 8-GPU 节点（HGX H200） | NVL72 机架（GB200） |
|---|---|---|
| 结构 | 8 张 GPU 通过 NVSwitch 全互联；8 张网卡，每张 400 Gb/s 连到 scale-out 网络 | 72 张 GPU 通过 NVLink switch 互联（每行 4 GPU + 2 CPU，共 9 行 × 2） |
| HBM（单卡） | 4.8 TB/s | 8 TB/s |
| NVLink（卡间） | 900 GB/s | 1.8 TB/s |
| RDMA（跨节点网络） | 50 GB/s | 100 GB/s |

- **Scale-up（纵向扩展）**：在一个 NVLink 域内（一台机器或一个机架）加更多 GPU，卡间带宽很高
- **Scale-out（横向扩展）**：通过网络（RDMA）连接更多机器，带宽低一个数量级以上

**带宽的层级**：HBM（数 TB/s）≫ NVLink（约 1–2 TB/s）≫ 网络（约 50–100 GB/s）。这决定了并行策略：通信频繁的 tensor parallelism 放在 NVLink 域内；通信较少的 pipeline / data parallelism 跨越网络。

### 2.9 进阶：NVIDIA GPU 的二十年

**2006–2017：从图形到通用计算**

| 架构 | 对程序员来说的变化 |
|---|---|
| Tesla（2006，G80） | 统一着色器；CUDA、thread block、shared memory |
| Fermi（2010，C2050） | 真正的 L1/L2 cache、ECC、统一地址空间、快速 FP64 |
| Kepler（2012，K20X） | warp shuffle、动态并行、Hyper-Q 工作队列 |
| Maxwell（2014，M40） | SM 分成四个分区；能效大幅提升 |
| Pascal（2016，P100） | HBM2、NVLink、支持缺页的统一内存、2 倍 FP16 |
| Volta（2017，V100） | **Tensor Core**、独立线程调度、统一的 L1/shared memory |

**2018–2026：从计算到低精度 AI 引擎**

| 架构 | 对程序员来说的变化 |
|---|---|
| Turing（2018，T4） | INT8 和 INT4 Tensor Core、光线追踪核心 |
| Ampere（2020，A100） | TF32、BF16、2:4 稀疏、cp.async、MIG、40 MB L2 |
| Hopper（2022，H100） | FP8、TMA、cluster 与 DSMEM、wgmma、机密计算 |
| Blackwell（2024，B200） | 双 die、FP6/FP4 与 block scaling、TMEM、tcgen05、NVL72 |
| Blackwell Ultra（2025，B300） | 288 GB HBM3e、15 PF dense NVFP4、更快的 exp、FP64 大幅减少 |
| Rubin（2026，NVL72） | HBM4、NVLink 6、更大的 TMEM 和 shared memory（初步） |

**算力远远跑在了内存前面**：从 G80 到 B300，峰值算力增长约 **43,000 倍**，带宽只增长约 **90 倍**。

---

## 3. 选读：其他加速器

### 3.1 AMD Instinct MI355X：一个封装里有更多 HBM 的 chiplet GPU

| 与 NVIDIA GPU 的不同 | 怎么编程 |
|---|---|
| 一个封装里有 8 个计算 die，软件上看是一张 GPU | ROCm 和 HIP：类似 CUDA 的 C++；hipify 可以转换 CUDA 源码 |
| HBM 前面有 256 MB 的 Infinity Cache | PyTorch、vLLM、SGLang 都能在 ROCm 上运行 |
| 64 线程的 wavefront，而不是 32 线程的 warp | 自定义 kernel 可以用 Triton 或 Composable Kernel |

规格：5.0 PF dense FP8（B200：4.5 PF）、288 GB HBM3E（B200：180 GB）、8 TB/s、scale-up 域 8 张卡（GB200 NVL72：72 张）。

### 3.2 Google TPU Ironwood：用 3D torus 连接的 systolic array

**SIMT lanes vs systolic array**：
- GPU 的 SIMT：**每条指令都要从寄存器文件读操作数、把结果写回去**
- **Systolic array（脉动阵列，weight-stationary 方式）**：权重**固定在每个单元中**；激活值从左边流进来；部分和向下流动、从底部流出。数据在相邻单元之间直接传递，**大大减少了对寄存器和内存的访问**，非常适合矩阵乘法

**怎么编程**：JAX 或 PyTorch/XLA；XLA 编译器把矩阵乘切块映射到 systolic array 上；用 Pallas 写自定义 kernel。

规格：4.6 PF dense FP8、192 GB HBM3E、7.4 TB/s、3D torus 最多连接 9,216 颗芯片。

### 3.3 Amazon Trainium3：systolic 引擎 + 软件管理的 SRAM

| 与 NVIDIA GPU 的不同 | 怎么编程 |
|---|---|
| 8 个 NeuronCore，每个有独立的 Tensor（systolic）、Vector、Scalar 引擎 | Neuron SDK 编译 PyTorch 和 JAX 的计算图 |
| 每个核心 32 MB SRAM，由 DMA 引擎填充；**没有硬件 cache** | NKI（Neuron Kernel Interface）：用 Python 写 kernel，**显式地**在 HBM 和 SRAM 之间搬运 tile |
| 性能提升主要来自 MXFP8 和 MXFP4；每核 BF16 与 Trainium2 相同 | 没有 CUDA；kernel 面向的是引擎，而不是线程 |

规格：2.5 PF dense MXFP8、144 GB HBM3E、4.9 TB/s、scale-up 域 144 颗芯片。

### 3.4 其他加速器：用片上 SRAM 代替 HBM

| 芯片 | 片上 SRAM | 片外内存 | 权衡 |
|---|---|---|---|
| Cerebras WSE-3 | 44 GB | 无 | 整片晶圆就是一颗芯片；更大的模型需要从外部流式加载权重 |
| Groq LPU | 230 MB | 无 | 一个 8-bit 的 70B 模型需要约 300 颗芯片 |
| Microsoft Maia 200 | 272 MB | 216 GB HBM3e | SRAM 存放热数据；只在 Azure 中可用 |

思路：SRAM 的带宽远高于 HBM，**decode 这种带宽受限的负载会快很多**；代价是容量小，需要很多芯片才能放下一个大模型。

---

## 4. 要点回顾

1. GPU 用成千上万个简单核心和大量常驻 warp 换取吞吐；warp 切换零开销，用来隐藏约 500 周期的内存延迟
2. SIMT：一条指令驱动一个 warp 的 32 个线程；分支发散会让 warp 轮流执行两条路径
3. 三项规格：算力（引擎）、带宽（油管）、容量（油箱）
4. Kernel 时间 ≈ max（字节 ÷ 带宽，FLOPs ÷ 峰值）；算术强度决定落在 roofline 的哪一侧；H200 的屋脊点约 206 FLOP/B
5. GEMV 和 decode attention 的算术强度约 1（BF16），极端 memory-bound；大 GEMM 是 compute-bound
6. Batch 1 的 decode 只用到约 0.5% 的算力；batching 帮助权重读取但不帮助 KV 读取；context 越长，KV 读取占比越大
7. 低精度让屋脊点右移，算力增长快于带宽，越来越多的 kernel 变成带宽受限
8. 内存层级：寄存器 → shared memory/L1 → L2 → HBM → NVLink / PCIe → 网络；高性能 kernel 的核心是在上层复用数据
9. Little's law：带宽 = 在途字节 ÷ 延迟；要跑满 H200 带宽，每个 SM 需要约 9 KB 在途；向量化读取、多个独立读取和 TMA 都在增加在途字节
10. Scale-up（NVLink）带宽远高于 scale-out（网络），决定了 TP 放在节点内、DP/PP 跨节点

---

## 5. 自测题

<details>
<summary>1. 一个 kernel 的算术强度是 50 FLOP/B，在 H200 上最多能达到多少 TFLOP/s？它是 compute-bound 还是 memory-bound？</summary>

min(989, 4.8 × 50) = 240 TFLOP/s。50 < 206（屋脊点），所以是 memory-bound。
</details>

<details>
<summary>2. 一个 70B 模型（BF16 权重 141 GB）在 2 张 H200（合计 9.6 TB/s）上做 batch 1 的 decode，理论上每秒最多生成多少 token？</summary>

每个 token 至少要读一遍权重：141 GB ÷ 9.6 TB/s ≈ 14.7 ms，所以最多约 68 tokens/s（还没算 KV cache 的读取）。
</details>

<details>
<summary>3. 为什么 batching 能提高 decode 的权重 GEMV 效率，却不能提高 decode attention 的效率？</summary>

所有请求共用同一份权重，一次读出可以被 batch 中每个请求复用，算术强度随 batch 增长；但每个请求有自己独立的 KV cache，读出的 KV 只服务这一个请求，算术强度不随 batch 改变。
</details>

<details>
<summary>4. 用 Little's law 解释：为什么一个每线程只发一次 4 字节读取的 kernel 跑不满 HBM 带宽？</summary>

带宽 = 在途字节 ÷ 延迟。延迟约 250 ns 是固定的，每个 SM 在途只有 32 warp × 32 线程 × 4 B = 4 KB，132 个 SM 合计约 540 KB，÷ 250 ns ≈ 2.2 TB/s，只有峰值的 45%。要跑满需要每 SM 约 9 KB 在途。
</details>

<details>
<summary>5. 一个 warp 中一半线程走 if 分支、一半走 else 分支，两条分支耗时相同。与没有分支发散相比，吞吐变成多少？</summary>

warp 要先执行 if 分支（一半 lane 闲置），再执行 else 分支（另一半闲置），总时间翻倍，有效吞吐约为一半。
</details>

<details>
<summary>6. 为什么低精度（FP8、FP4）让屋脊点右移？这对 kernel 设计意味着什么？</summary>

低精度下峰值算力成倍提高，而 HBM 带宽不变，屋脊点 = 峰值 ÷ 带宽随之变大。kernel 需要更高的算术强度（每字节更多运算）才能 compute-bound，越来越多的 kernel 会受带宽限制，所以减少数据搬运（复用、融合、压缩）变得更重要。
</details>

<details>
<summary>7. Systolic array 为什么适合矩阵乘法？</summary>

权重固定在各个单元中，激活值和部分和在相邻单元之间直接流动，每个数据被多次复用，而不需要每次运算都读写寄存器文件或内存，大大减少了数据搬运，这正是矩阵乘法的需求。
</details>
