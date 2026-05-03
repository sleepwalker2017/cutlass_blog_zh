---
title_zh: "CUTLASS 教程：在 NVIDIA® Hopper™ GPU 上使用 WGMMA 进行快速矩阵乘法"
title_en: "CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper™ GPUs"
source_url: "https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/"
published_at: "2024-08-06"
source_site: "Colfax Research"
external: false
english_markdown: "articles-en/cutlass-tutorial-wgmma-hopper.en.md"
---
# CUTLASS 教程：在 NVIDIA® Hopper™ GPU 上使用 WGMMA 进行快速矩阵乘法

原文标题：CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper™ GPUs

英文对照：[articles-en/cutlass-tutorial-wgmma-hopper.en.md](../articles-en/cutlass-tutorial-wgmma-hopper.en.md)

如果一个 CUDA® 教程系列里没有 GEMM（通用矩阵乘法），那几乎是不完整的。GEMM 可以说是现代 GPU 上最重要的基础例程之一：神经网络、大语言模型，以及大量图形应用中的主要计算，都建立在它之上。尽管 GEMM 无处不在，但要把它写得高效并不容易。

这个由 3 部分组成的教程系列，旨在帮助读者系统理解如何使用 CUTLASS 在 NVIDIA Hopper GPU 上编写高效的 GEMM 内核。

- [第 1 部分（也就是本文）] 讨论 warpgroup 矩阵乘加（WGMMA）指令，它们是面向 Hopper 架构 NVIDIA GPU 上 Tensor Core 的底层原语。
- [第 2 部分](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)讨论高效 GEMM 内核的整体设计，包括 CUTLASS 内核中使用的一些高级技术，例如 warp specialization 和 ping-pong 调度。
- [第 3 部分] 讨论 persistent kernel 与 [Stream-K](https://arxiv.org/abs/2301.03598)。它们是 GEMM 中的重要负载均衡策略，能够在大量问题形状上实现很高的效率。

**整体脉络** 本系列共 3 篇，基本对应 GEMM 内核从底层到上层的三个抽象层次。第一层是 tile 级 GEMM 原语，即直接驱动 Tensor Core 执行计算的那一层。第二层是单个 CTA 视角下的内核设计，通常由 *prologue*、*mainloop* 和 *epilogue* 组成；这一层的核心是让计算与数据搬运充分重叠，避免 Tensor Core 因内存访问而空转。第三层是网格级 CTA 调度；在这一层，关键问题是负载均衡。

我们希望，读完这个系列之后，读者不仅能真正理解 GEMM 的实现方式，也能把其中一些漂亮的设计思想迁移到自己的内核开发工作里。

### 异步 warpgroup MMA（WGMMA）

Hopper 引入了异步的 warpgroup 级矩阵乘加操作（WGMMA）。一个 *warpgroup* 由 4 个连续的 warp 组成，也就是 128 个连续线程；其中第一个 warp 的编号必须是 4 的倍数。`wgmma.mma_async` 指令由一个 warpgroup 中全部 128 个线程共同执行。它通常具有以下两种形式之一，其中矩阵 `C` 作为累加器：

- `C = A * B + C`
- `C = A * B`，其中累加器 `C` 的输入被禁用。

WGMMA 有一个重要限制：操作数 `B` 必须始终存放在共享内存（SMEM）中。相比之下，操作数 `A` 可以位于 SMEM，也可以位于寄存器内存（RMEM）中；而累加器 `C` 始终保存在 RMEM 中。

本文结构如下。首先，我们讨论在 CUTLASS 中调用 `wgmma.mma_async` 的关键点，这包括构建相应的 `TiledMMA` 对象，以及创建并分区与 WGMMA 兼容的 SMEM 张量。其次，我们讨论保证 WGMMA 正确性所需的同步机制。最后，我们会更深入地解释 WGMMA 中使用的布局，包括所谓的 *core matrix* 与 *matrix descriptor* 等概念，它们都与源自 SMEM 的操作数有关。

为了简洁起见，后文会把 `wgmma.mma_async` 简写为 `wgmma`。本文主要参考的代码，是 CUTLASS 的 [wgmma 教程](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/examples/cute/tutorial/wgmma_sm90.cu)，由 Pradeep Ramani 编写，并在 3.5.1 版本中加入。

### CUTLASS 内核中的 WGMMA

本教程的主要目标，是解释 `wgmma` 这组原语如何调用 Hopper 上的 Tensor Core 来执行基于 tile 的 GEMM，以及它们如何被封装进 `cute::gemm` 调用中。为了说明问题，我们先考虑一个标准的 GEMM 内核：输入矩阵 `A` 和 `B` 的尺寸为 `MxNxK`，输出满足 `C = A * B`。为了并行化计算，内核会固定静态 tile 尺寸 `bM`、`bN` 和 `bK`，并启动一个 `⌈M/bM⌉ x ⌈N/bN⌉` 的 CTA 网格，其中每个 CTA 负责输出矩阵中的一个 `bM x bN` tile `rC`。这个结果会先保存在 CTA 的 RMEM 中，最后再写回全局内存中的 `C`。

从单个 CTA 的角度看，内核的核心是它的 *mainloop*。在 `⌈K/bK⌉` 次迭代中，我们沿着内部维度循环，依次把 `A` 和 `B` 的 `bM x bK` 与 `bN x bK` tile 从全局内存加载到共享内存中的 `sA` 与 `sB`。需要注意的是，在 CUTLASS 中，`sB` 的布局会被组织成数学意义上的转置形式。实际上，按照常见做法，这些 tile 会被加载进循环使用的 SMEM 缓冲区，阶段数通常在编译期固定为 2 或 3，因此 `sA` 和 `sB` 的形状元组最后一个模式就是这个 stage 数。随后，`cute::gemm` 会对 `sA` 和 `sB` 的相应 stage 切片做乘加，并把结果持续累加到 `rC` 中。主循环结束后，epilogue 再把 `rC` 写回全局内存。

下面我们来解释 `cute::gemm` 调用本身，以及它依赖的那些参数。下面这段代码摘自 [wgmma 教程](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/wgmma_sm90.cu#L73)，其中省略了与本文无关的部分，例如流水线化的 TMA 加载：

```

template <class TiledMMA, ... >
__global__ device_gemm(TiledMMA tiled_mma, ...) {
  // PROLOGUE
  // ...
  // Define A/B partitioning and C accumulators
  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // Allocate "fragments"
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)
  
  // PIPELINED MAIN LOOP
  while (k_tile_count > -K_PIPE_MAX) {
    // ...
    // MMAs to cover 1 K_TILE
    cute::warpgroup_arrive();
    // (V,M,K) x (V,N,K) => (V,M,N)
    cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    cute::warpgroup_commit_batch();
    // Wait for all MMAs in a K_TILE to complete
    cute::warpgroup_wait<0>();
    // ...
  }

  // EPILOGUE
  // ...
}
```

在 CUTLASS 的 [MMA 范式](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md)中，`cute::gemm` 的目标，是用统一接口来暴露不同架构下的 MMA 指令。事实上，如果你去看 [SM80 教程里的 GEMM 内核](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu#L275)，会发现那里的 `cute::gemm` 在语法上和这里几乎一模一样。不过，WGMMA 场景下与 `cute::gemm` 相关的参数定义，带有一些明显的 WGMMA 特性：

- `TiledMMA` 对象 `tiled_mma` 的定义，封装了 `cute::gemm` 最终发射特定 `wgmma` PTX 指令所需的信息。
- SMEM 张量 `sA` 和 `sB` 的布局必须满足 `wgmma` 的兼容约束。
- `tCrA`、`tCrB` 和 `tCrC` 这些 fragment，是通过 `TiledMMA` 的线程级分区方法构造的，因此其布局具有 WGMMA 特有属性。
- `tCrA`（当操作数 `A` 来自 SMEM 时）和 `tCrB` 并不是“把 SMEM 值拷到寄存器”得到的常规寄存器张量，而是基于 SMEM 构造的矩阵描述符视图。

最后，还需要解释 `cute::gemm` 前后包裹的同步调用。下面会按顺序展开。

### WGMMA 的 TiledMMA 对象

接下来，假设数据类型为 FP16，且 `A` 和 `B` 都是 `MN`-major（按 BLAS 记号即 NT GEMM）。我们在主机侧通过 `cute::make_tiled_mma` 构造 `TiledMMA` 对象：

```

TiledMMA tiled_mma = cute::make_tiled_mma(
  SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});
```

`cute::make_tiled_mma` 还有一些可选参数，这里先聚焦最核心的 *MMA atom*。它是对底层 PTX 调用的封装，在本例中对应：

```

wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
```

CUTLASS 的命名可以直接映射到 PTX 指令语义。首先，SM90 对应 Hopper 架构。SM90 MMA atom 一般命名为 `SM90_MxNxK_XYZ_SS` 或 `SM90_MxNxK_XYZ_RS`，并带有两个模板参数（`GMMA::Major::MN` 或 `GMMA::Major::K`）。含义如下：

- `X`和`Y`是操作数的数据类型。
- `Z`是累加器的数据类型。
- `MxNxK` 是 `wgmma` 指令计算时使用的 tile 尺寸。它并非任意取值，可选范围见[官方列表](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape)：`M` 固定为 64，`N` 是 8 到 256 的 8 倍数；对 16 位操作数，`K` 为 16（更一般地说，`K` 固定为 32 字节）。
- 后缀 `RS` 或 `SS` 表示操作数 `A` 来自寄存器（`R`）还是共享内存（`S`）。操作数 `B` 始终来自共享内存，因此一定是 `S`。
- 两个模板参数表示操作数 `A` 与 `B` 在内存中是按 `MN` 模式连续还是按 `K` 模式连续。例如按 BLAS 记号，两者都是 `K`-major 对应 TN GEMM（见[这张表](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md#aside-m-major-n-major-k-major)）。注意：对 16 位操作数，`MN`-major 与 `K`-major 都可用；但对非 16 位操作数，**布局必须是 `K`-major**。

以上就是 MMA atom 语法的核心。接下来强调一点：WGMMA 是 *warpgroup 范围* 指令。代码中可通过 `TiledMMA` 查询参与该 MMA 操作的线程数（`size`）。例如：

```

dim3 dimBlock(cute::size(tiled_mma));
```

这表示内核中的每个 CTA 以 1 个 128 线程的 warpgroup 启动。

假设我们希望由 *2 个 warpgroup* 执行 WGMMA，让两个 warpgroup 分别计算输出 tile 的一半（每个 warpgroup 发射各自的 `wgmma` 指令）。这时可把非平凡布局 `AtomLayoutMNK` 作为 `make_tiled_mma` 的第二个参数。例如：

```

 TiledMMA tiled_mma = make_tiled_mma(
  SM90_64x64x16_F16F16F16_SS{},
  Layout<Shape<_2,_1,_1>>{});
```

这定义了一个 WGMMA 操作：warpgroup 1 和 2 分别计算输出 tile 的上半与下半，并沿 `M` 模式切分（此时假设 `bM` 是 128 的倍数）。同时，`size(tiled_mma)` 会变为 256。

一般来说，`make_tiled_mma` 的两个可选布局参数 `AtomLayoutMNK` 和 `PermutationMNK`，对任意 MMA atom 的作用机制是一致的。关于 `PermutationMNK` 的用途，推荐阅读 Cris Cecka 的[解释](https://github.com/NVIDIA/cutlass/discussions/1345)。

### SMEM WGMMA 的布局约束

接下来，我们解释在选择 MMA 原子的情况下，SMEM 中操作数矩阵的tile 大小和布局的约束。首先，对于任何 MMA 指令，`MxNxK`MMA 原子的原子需要分为操作数和累加器块的原子。在我们的例子中，这意味着`bM`应该是 64 的倍数，`bN`64 的倍数，以及`bK`16的倍数。

其次，WGMMA 对 SMEM 布局有额外约束：`sA` 和 `sB`（形状与步幅）必须满足所选 swizzle 模式的要求。特别是，`sA`（以及 `sB`）的每个 stage 切片通常不能直接用简单布局 `(bM,bK):(1,bM)` 或 `(bM,bK):(bK,1)` 表示。

要深入理解这些要求，需要以下概念：*核心矩阵*，下面我们就来介绍一下。然而，实际上，我们总是可以构建保证兼容的布局`wgmma`使用 CUTLASS 提供的某些预定义布局原子，然后是`cute::tile_to_shape`方法。在我们的示例中，我们准备了tile尺寸和`sA`, `sB`在主机上如下（与`T=cutlass::half_t`这是 CUTLASS 对 FP16 的名称）：

```

auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};  
auto bP = Int<  3>{};  // Pipeline

auto sA = cute::tile_to_shape(
	GMMA::Layout_MN_SW128_Atom<T>{},
	cute::make_shape(bM, bK, bP)
);
auto sB = cute::tile_to_shape(
	GMMA::Layout_MN_SW128_Atom<T>{},
	cute::make_shape(bN, bK, bP)
);
```

这里，`MN`表示布局原子适合`MN`-主操作数，和`SW128`是128字节的swizzle模式。打印输出`sA`或者`sB`显示

```

Sw&lt;3,4,3> o smem_ptr[16b](unset) o ((_64,_2),(_8,_8),_3):((_1,_512),(_64,_1024),_8192)
```

这个布局怎么来的？`cute::tile_to_shape` 会把布局 atom 平铺到更大的目标形状（类似 `numpy.tile`）。先忽略 swizzle 函数 `Sw<3,4,3>`，布局 atom 可写作 `(64,8):(1,64)`，并被平铺到 `(128,64,3)`。对 `MxK` 而言，外层较小步幅 `512` 对应 `M` 模式，较大步幅 `1024` 对应 `K` 模式；最大步幅 `8192` 对应 stage 维 `P`，这保证不同 stage 切片不会在内存中互相重叠。

注意`64`次`sizeof(half_t)`等于 128 字节，这是 swizzle 模式的名称。这是设计使然：由于核心矩阵的工作方式，我们总是将连续方向上的布局原子的长度安排为等于 swizzle 字节数 - 或者`16`不调酒，或其中之一`32`, `64`， 或者`128`.

相反，如果我们考虑：

```

auto sA = cute::tile_to_shape(
  GMMA::Layout_K_SW128_Atom<T>{},
  cute::make_shape(bM,bK,bP)
);
auto sB = cute::tile_to_shape(
  GMMA::Layout_K_SW128_Atom<T>{},
  cute::make_shape(bN,bK,bP)
);
```

然后打印`sA`会给我们

```

Sw&lt;3,4,3> o smem_ptr[16b](unset) o (_128,_64,_3):(_64,_1,_8192)
```

因为我们改为平铺`(8,64):(64,1)`超过`(128,64,3)`。 （注意布局`((_8,_16),(_64,_1),_3):((_64,_512),(_1,_0),_8192)`合并为`(_128,_64,_3):(_64,_1,_8192)`).

一般来说，可选 8 种布局 atom：对应 `MN` 或 `K` major，再乘以 4 种 swizzle 模式。

- 无 swizzle：默认 16 字节边界。
- 32B swizzle：交错 2 个连续 16B 段。
- 64B swizzle：交错 4 个连续 16B 段。
- 128B swizzle：交错 8 个连续 16B 段。

布局原子被定义[这里](https://github.com/NVIDIA/cutlass/blob/36cbfcf483cc9d2ee65a55c199176ce96da1e33e/include/cute/atom/mma_traits_sm90_gmma.hpp#L66)在 CUTLASS 代码库中为：

```

GMMA::Layout_MN_INTER_Atom<T>
GMMA::Layout_MN_SW32_Atom<T>
GMMA::Layout_MN_SW64_Atom<T>
GMMA::Layout_MN_SW128_Atom<T>

GMMA::Layout_K_INTER_Atom<T>
GMMA::Layout_K_SW32_Atom<T>
GMMA::Layout_K_SW64_Atom<T>
GMMA::Layout_K_SW128_Atom<T>
```

随后把这些布局 atom 传给 `tile_to_shape`，目标形状分别是 `make_shape(bM,bK,bP)`（给 `sA`）和 `make_shape(bN,bK,bP)`（给 `sB`）。注意 mode 顺序必须保持一致，且 layout atom 的平铺尺寸必须整除目标 SMEM 形状。这些是 swizzle 选择带来的约束，和 MMA atom 形状本身的约束是独立的。

### WGMMA 片段和描述符

至此我们已经创建了 `TiledMMA`，并在主机侧准备好 SMEM 布局。设备侧可用 `tiled_mma` 构造传给 `cute::gemm` 的分区张量。首先，通过 `tiled_mma.get_thread_slice(threadIdx.x)` 创建 `ThrMMA` 对象 `thr_mma`；在本例中线程索引范围是 `0..127`。

然后，参考上面的内核代码片段，打印张量`tCsA`和`tCsB` **对于任何线程索引**显示以下内容：

```

tCsA: Sw&lt;3,4,3>_smem_ptr[16b](0x7f8800000400) o
	((_64,(_8,_2)),_2,_4,_3):((_1,(_64,_1024)),_512,_2048,_8192)
tCsB: Sw&lt;3,4,3>_smem_ptr[16b](0x7f880000c400) o
	((_64,(_8,_2)),_2,_4,_3):((_1,(_64,_1024)),_512,_2048,_8192)
```

根据评论，形状`tCsA`应该被认为是`(MMA,MMA_M,MMA_K,PIPE)`:

- `MMA`是`NxK`MMA Atom 的形状。
- `MMA_M`和`MMA_K`是其平铺的范围`M`和`K`的模式`sA`（以便`MMA_M=bM/64=2`和`MMA_K=bK/16=4`).
- `PIPE`是阶段数。

其步幅和 swizzle 模式继承自 `sA`。这里的 WGMMA 特性在于：`tCsA` 并不是真正意义上的“线程私有 SMEM 切片”，而是一个按 WGMMA 规则重组后的 SMEM 视图。

接下来，打印“片段”`tCrA`和`tCrB`对于任何线程索引显示：

```

tCrA: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
tCrB: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
```

在内部，CUTLASS 构造了一个“[矩阵描述符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)“，这是保存在寄存器中的 64 位值，以适合使用的方式描述 SMEM`wgmma`操作说明。对于程序员来说，最重要的是要记住 SMEM 的值是**不是**复制到RMEM；相反，访问的值`tCrA`和`tCrB`相反，访问这些 64 位描述符。此外，这些张量是“迭代器”，意味着只有单个 64 位描述符用于给定的`wgmma`指令一次保存在寄存器中（例如，而不是全部 24 个）。

与操作数相比，累加器张量的定义更“常规”。对线程 0 打印 `tCgC` 和 `tCrC` 可见：

```

tCgC: gmem_ptr[16b](0x7f877a780000) o ((_2,_2,_8),_2,_2):((512,_8,4096),_64,32768)
tCrC: ptr[16b](0x7feee1fffbe0) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)
```

`tCgC` 是输出 GMEM 张量切片（epilogue 阶段会把累加器结果写回这里）；`tCrC` 是寄存器支撑的张量，用于保存 mainloop 中累加得到的值。它们的形状 `(MMA, MMA_M, MMA_N)` 可这样理解：在 `MxN = 64x64` 的 MMA atom 输出 tile 中，128 个线程每个保存 `32 = 2*2*8` 个值；并且 `MMA_M = MMA_N = 2`，与 `tCsA`/`tCsB` 对应。

每个线程保存的 32 个值之所以拆成 `(2,2,8)`，是为了与 `tCgC` 的布局步幅定义对齐。更具体的分配模式可参考 [PTX 文档中的图示](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d)：

![](../images/cutlass-tutorial-wgmma-hopper/wgmma-64N16-D-1-e28ffdbdf8.png)

它展示了复制中的 Z pattern，即线程持有 32 个值的分布方式。比如线程 0 保存 `(0,0)`、`(0,1)`、`(8,0)`、`(8,1)` 等位置的值，并沿列方向每 8 列重复。

### 重新审视 gemm 调用

让我们回到上面内核代码片段的第 25 行：

```

// (V,M,K) x (V,N,K) => (V,M,N)
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
```

`cute::gemm` 的多层重载会先遍历外层 mode（`MMA_M/N` 和 `MMA_K`）。当这些坐标固定后，内部就退化为对 MMA atom tile 的计算。换句话说，先把高层 `cute::gemm` 降到[调度形状](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/algorithm/gemm.hpp#L178) `(V)x(V)=>(V)`。

随后会调用 MMA atom 的 [`fma` 操作](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/arch/mma_sm90_gmma.hpp#L401)（更准确地说，经由 [`mma_unpack`](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/atom/mma_traits.hpp#L112)）。其中包含如下内联 PTX：

```

CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t& d00, uint32_t& d01, uint32_t& d02, uint32_t& d03,
      uint32_t& d04, uint32_t& d05, uint32_t& d06, uint32_t& d07,
      uint32_t& d08, uint32_t& d09, uint32_t& d10, uint32_t& d11,
      uint32_t& d12, uint32_t& d13, uint32_t& d14, uint32_t& d15,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15},"
      " %16,"
      " %17,"
      " p,   %19, %20, %21, %22;\n"
    "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03),
        "+r"(d04), "+r"(d05), "+r"(d06), "+r"(d07),
        "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15)
      : "l"(desc_a),
        "l"(desc_b),
        "r"(int32_t(scale_D)),
        "n"(int32_t(scaleA)),
        "n"(int32_t(scaleB)),
        "n"(int32_t(tnspA)),
        "n"(int32_t(tnspB)));
#else
    CUTE_INVALID_CONTROL_PATH(
    	"Attempting to use SM90_64x64x16_F16F16F16_SS " 
    	"without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
  }
```

这段语法对应的 PTX 说明在[这里](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)。与上文 `tCrA`、`tCrB`、`tCrC` 的解释一致：操作数通过 `uint64` 的 `desc_a`/`desc_b` 传入，累加器由 16 个 `uint32` 变量承载。`scale_D` 取 `0` 或 `1`，控制是否对累加器做零初始化。

此外，`scaleA`、`scaleB`、`tnspA`、`tnspB` 这些量通过模板参数在编译期确定。`scaleA`/`scaleB` 取 1 或 -1（用于符号翻转），`tnspA`/`tnspB` 表示是否转置操作数，其值由 `GMMA::Major::K` 或 `GMMA::Major::MN` 决定。

### WGMMA 的同步

还需要解释 `cute::gemm` 前后的这组调用：

```

cute::warpgroup_arrive();
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
cute::warpgroup_commit_batch();
cute::warpgroup_wait<0>();
```

为什么要有这些额外调用？因为 `wgmma` 本质上是 *异步* 指令。在 Hopper 上，“异步”意味着 `wgmma` 可与其他操作并行推进，因此依赖关系必须用显式同步来约束。其机制可参考 PTX 的[内存一致性模型](https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html#program-order-async-operations)。同步写错会导致：(a) 隐蔽竞态；(b) 编译器把 `wgmma` 串行化从而显著降速；(c) 未定义行为。

突出显示的`cute`方法包装以下 PTX 指令：

- `cute::warpgroup_arrive()` — `wgmma.fence.sync.aligned`;
- `cute::warpgroup_commit_batch()` — `wgmma.commit_group.sync.aligned`;
- `cute::warpgroup_wait<N>()` — `wgmma.wait_group.sync.aligned N`;

（注意：文中一直将 `wgmma.mma_async` 简写为 `wgmma`。）下面把这些命令与 PTX 文档中 WGMMA GEMM 的流程对应起来：

1. 将矩阵 `A`、`B`、`D` 加载到寄存器或共享内存。
2. 执行 `fence`：`wgmma.fence` 用于声明 warpgroup 视角下的 register/shared-memory 写入已就绪；`fence.proxy.async` 用于让通用代理写入对异步代理可见。
3. 通过 `wgmma.mma_async` 发射异步 MMA 运算（在异步代理中执行）。
4. 通过 `wgmma.commit_group` 创建并提交一个 wgmma-group，把此前尚未提交的 `wgmma.mma_async` 纳入该组。
5. 通过 `wgmma.wait_group` 等待所需 wgmma-group 完成。
6. 组完成后，该组内所有 `wgmma.mma_async` 均已执行完毕。

先看 `wgmma.fence`：它保证 `wgmma.mma_async` 访问某些 RMEM 地址前，相关先前访问已经完成。缺少 `wgmma.fence` 会导致未定义行为。一个例外是 Hopper 允许多个 `wgmma.mma_async` 在 flight；若它们累加器形状一致，可共享同一累加器张量（即写同一批寄存器地址），这种情况下不需要额外 fence。例如在 `cute::gemm` 内部沿 `MMA_K` 循环时，通常不必每步插入 `wgmma.fence`。

与 [TMA 操作](https://research.colfax-intl.com/tutorial-hopper-tma/)类似，`wgmma.mma_async` 运行在[异步代理](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy)中。因此，*如果* 通用代理中的操作会影响 `wgmma.mma_async` 读取的 SMEM 内容，就需要 `fence.proxy.async`。例如使用普通 `ld.global / st.shared` 把 `A/B` 搬到 SMEM 时就属于这种情况。本文示例使用的是 TMA load，所以不需要 `fence.proxy.async`；这也是它在教程代码与 CUTLASS Hopper GEMM mainloop 中都未出现的原因（对应封装是 `cutlass::arch::fence_view_async_shared()`）。

`wgmma.commit_group` 会为当前 warpgroup 创建一个新的 wgmma-group，并把该 warpgroup 先前发起但尚未归组的 `wgmma.mma_async` 一次性提交到该组。在我们的示例中，`cute::warpgroup_commit_batch()` 会把 `MMA_M*MMA_N*MMA_K` 条 `wgmma.mma_async` 放入同一组。

最后，`wgmma.wait_group N` 会让执行线程等待，直到“最多只剩 `N` 个最新 wgmma-group 仍在 pending”，并保证更早提交的组已完成。在我们的例子里 `N=0`，因此 warpgroup 会等待当前组全部完成后再执行后续指令。

在 warpgroup 有机会执行独立计算的情况下，参数的灵活性`N`派上用场了。例如，这与设计中采用的 GEMM-softmax 重叠策略一起发挥作用[FlashAttention-3](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/).

### WGMMA 核心矩阵

最后一节继续讨论 `A`、`B` 加载到 SMEM 后的布局约束（假设 `wgmma` 两个操作数都来自 SMEM）。为简化说明，先假设 `A` 为行主、`B` 为列主（即都可视作 `K`-major）。回忆 `wgmma` 的 tile 形状 `MxNxK` 受限：`M=64`，`K * sizeof(dtype)=32B`，`N` 是 8 到 256 的 8 倍数。为避免与 `A/B` 或 `sA/sB` 混淆，这里将 WGMMA atom tile 记作 `wA` 与 `wB`。

矩阵 `wA` 和 `wB` 会被分解为更小的 *core matrix*。每个 core matrix 有一个 stride 方向和一个 contiguous 方向：stride 方向长度为 8，contiguous 方向长度为 16 字节。`wA` 由 `8x2` 个 core matrix 组成，`wB` 由 `2x(N/8)` 个 core matrix 组成。其平铺方式如下（图来自 PTX 文档）：

![](../images/cutlass-tutorial-wgmma-hopper/wgmma2-b1903ccf45.png)

![](../images/cutlass-tutorial-wgmma-hopper/wgmma3-bcb36fa6cc.png)

如上，`wgmma` 在 `SS` 模式下需要为 `wA`（`desc-a`）和 `wB`（`desc-b`）提供[矩阵描述符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)。描述符编码以下 5 个参数：

- 起始地址：SMEM 中操作数的起始基地址。
- LBO (*主维字节偏移量*): 中两个相邻核心矩阵之间的距离（以字节为单位）`K`方面。
- SBO (*步幅维度字节偏移量*): 中两个相邻核心矩阵之间的距离（以字节为单位）`M`或者`N`方面。
- 调配模式：无、32、64 或 128 字节。
- Matrix base offset：用于处理 SMEM 地址未对齐到 swizzle 周期边界时的对齐问题。

LBO 和 SBO 如上图所示。

这[`make_gmma_desc`](https://github.com/NVIDIA/cutlass/blob/06b21349bcf6ddf6a1686a47a137ad1446579db9/include/cute/atom/mma_traits_sm90_gmma.hpp#L194C1-L194C54)CUTLASS 中的方法构造描述符（作为`GmmaDescriptor`）基于作为输入提供的 SMEM 张量布局。假设输入张量的布局是使用八个规范 GMMA 布局原子之一创建的，并且`tile_to_shape`，如先前在“SMEM WGMMA 的布局约束”中详细介绍的，`make_gmma_desc`将准确计算LBO和SBO，确定swizzling模式，并构造描述符。例如，`GmmaDescriptor`描述了以下允许的 WGMMA 布局`K`-主要情况（其中`T*sizeof(dtype)=16`):

```

No swizzle       : Swizzle&lt;0,4,3> o smem_ptr o ((8,m),(T,2)):((1T,SBO),(1,LBO))
32-byte swizzle  : Swizzle&lt;1,4,3> o smem_ptr o ((8,m),(T,2)):((2T,SBO),(1, T ))
64-byte swizzle  : Swizzle&lt;2,4,3> o smem_ptr o ((8,m),(T,2)):((4T,SBO),(1, T ))
128-byte swizzle : Swizzle&lt;3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1, T ))
```

对于由 GMMA layout atom 经 `tile_to_shape` 生成的[*紧凑（compact）*](https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/layout.hpp#L415)布局（注意：在 64B/128B swizzle 场景下，GMMA 的 `K`-layout atom 在 `K` 方向会大于 WGMMA atom 形状），对应 LBO/SBO 为：

```

No swizzle       : LBO = 16x8 = 128 bytes. SBO = 32x8 = 256 bytes.
32-byte swizzle  : SBO = 32x8 = 256 bytes.
64-byte swizzle  : SBO = 64x8 = 512 bytes.
128-byte swizzle : SBO = 128x8 = 1024 bytes.
```

最关键的是：在 64B 和 128B swizzle 下，可接受的 WGMMA 布局通常**不是 compact**。可以理解为有 2 或 4 组 WGMMA atom 操作数块并排堆在 `K` 方向，从而使 core matrix 在 `M` 方向出现 `4T` 或 `8T` 的大步幅。也就是说，内存中逻辑上在 `K` 方向相邻的 2/4/8 个 core matrix，实际上会落到不同的 WGMMA atom 中。

为了完整起见，我们还给出了可接受的 WGMMA 布局`MN`-重大案件：

```

No swizzle       : Swizzle&lt;0,4,3> o smem_ptr o ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
32-byte swizzle  : Swizzle&lt;1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
64-byte swizzle  : Swizzle&lt;2,4,3> o smem_ptr o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
128-byte swizzle : Swizzle&lt;3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
```

### 结论

在 GEMM 系列的[第 1 部分] 中，我们介绍了在 Hopper GEMM 中把 WGMMA（warpgroup matrix multiply-accumulate）作为底层原语时需要掌握的核心概念。

WGMMA 需要一个 warpgroup（128 线程）协同执行，并且只能作用在受约束的矩阵分块上。我们分析了其中涉及的特殊 shape 与 layout，重点说明了如何通过标准 GMMA layout atom + `tile_to_shape` 构造出可被 WGMMA 接受的操作数布局。

为了明确其用法，WGMMA 还需要某些同步机制。为此，我们解释了`wgmma.fence`, `fence.proxy.async`, `wgmma.commit_group`和`wgmma.wait_group`关于`wgmma.mma_async`.

最后，我们详细解释了 WGMMA 核心矩阵的内部工作原理，以及 CUTLASS 如何为来自 SMEM 的操作数构造矩阵描述符。

总之，本文旨在帮助你在 Hopper 上使用 WGMMA 编写 CUTLASS 内核。在[第 2 部分]中，我们会进一步引入 TMA，并讨论如何在 Hopper GEMM 内核中把 TMA 与 WGMMA 串联起来，实现 copy 与 compute 的重叠。
