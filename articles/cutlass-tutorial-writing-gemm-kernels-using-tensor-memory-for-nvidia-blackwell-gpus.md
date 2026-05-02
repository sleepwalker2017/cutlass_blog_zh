---
title_zh: "CUTLASS 教程：使用张量内存为 NVIDIA® Blackwell GPU 编写 GEMM 内核"
title_en: "CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA® Blackwell GPUs"
source_url: "https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/"
published_at: "2025-04-19"
source_site: "Colfax Research"
external: false
english_markdown: "articles-en/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus.en.md"
---
# CUTLASS 教程：使用张量内存为 NVIDIA® Blackwell GPU 编写 GEMM 内核

原文标题：CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA® Blackwell GPUs

英文对照：[articles-en/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus.en.md](../articles-en/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus.en.md)

NVIDIA Blackwell 架构引入了一些新功能，这些功能显着改变了 GEMM 内核的形状。在本系列文章中，我们探索 Blackwell 上可用的新功能，并通过借鉴 CuTe 教程示例来研究如何编写利用这些新功能的 CUTLASS GEMM 内核。

- [第 1 部分，这一篇]讨论特定于 Blackwell 的第五代 Tensor Core MMA 指令，以及它们使用的张量内存。
- [第 2 部分] 解释了如何使用集群，包括围绕 TMA 多播和 Blackwell 的 CTA 对概念的新注意事项。
- [第 3 部分] 描述了具有较低精度数据类型的 MMA，以及 Blackwell 如何原生支持 MMA 的块缩放。

本系列的目标是解释如何更新 Hopper GEMM 内核以在 Blackwell 架构上运行，或者从头开始编写 Blackwell GEMM 内核。

在本文中，我们将介绍 Blackwell 的 MMA 指令和张量内存。我们将从这两个功能的总体摘要开始，然后再继续对这些功能进行 CUTLASS 抽象。然后我们将学习第一个[CuTe Blackwell 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/blackwell)重点关注自 Hopper 以来发生的变化。这篇文章的目的是解释使用 Blackwell 的新功能的简单 GEMM 内核的最小工作示例。

请注意，消费者 Blackwell 架构（计算能力 12.0）与数据中心 Blackwell 架构（计算能力 10.0）在一些主要方面有所不同，特别是缺少 Tensor Memory。我们将在这些帖子中仅讨论数据中心 Blackwell。

如果您尝试在 Blackwell GPU 上运行 CUTLASS Hopper GEMM 内核，您首先会注意到它不起作用。 Hopper WGMMA 指令（在 PTX 中，`wgmma.mma_async`) 已在 Blackwell 上弃用。为了取代它，Blackwell 引入了`tcgen05.mma`MMA 的指令。在CUTLASS中，`tcgen05.mma`被称为**UMMA**，为了简洁起见，我们将在以后采用这个术语。该新指令旨在替换 Hopper 中的 WGMMA。就像 WGMMA 一样，UMMA 是一条异步指令，用于计算以下矩阵运算之一：

`D = A * B + DD = A * B`

然而，与WGMMA相比，存在一些重大差异。

- 支持低精度数据类型，包括 FP4 和 FP6，并提高了所有精度的吞吐量。
- 内置块缩放支持。
- Tensor Core 称为 Tensor Memory 的专用存储器，用于 UMMA 累加。
- SM 簇内的两个相邻 CTA，称为**CTA一对**，可以跨两个 SM 一起在 UMMA 上工作。
- 与WGMMA不同，仅使用一个线程来启动UMMA。即使使用两个 CTA，一个 CTA 中也只有一个线程启动 UMMA。

在本文中，我们将主要关注第三点，讨论 Tensor Memory：它是什么以及如何将它用于 UMMA。

## 张量记忆

**张量内存（TMEM）**是供Tensor Cores使用的专用片上存储器。其主要目的是使用 TMEM 来替换第五代 Tensor Core 操作的寄存器。特别是对于 UMMA，该指令需要以下输入源：

- 操作数 A 可以是 TMEM 或 SMEM
- 操作数 B 必须位于 SMEM 中
- 累加器必须位于 TMEM

这意味着UMMA不需要寄存器来存储数据，减少了MMA操作的寄存器压力。此外，由于缺乏寄存器要求，加上单线程启动，可以进一步将 MMA 与 CTA 的主执行解耦。与 TMA 结合，CTA 在标准 GEMM 中直接执行的唯一处理是预处理和后处理。

在历史背景下，这些发展延续了用专用硬件资源取代通用计算资源的趋势，以消除瓶颈并释放这些通用资源用于其他操作。从 Volta 架构开始，Tensor Cores 将 GEMM 算术运算与通用计算管道分离。Ampere 的异步复制指令可实现 GEMM 主循环的真正流水线操作。在 Hopper GPU 上，异步、单线程 TMA 以及在 warpgroup 之间重新分配寄存器的能力极大地降低了数据移动的寄存器和线程成本，并且异步 WGMMA 允许 MMA 与其他计算操作进行流水线操作。现在，Tensor Memory 和 UMMA 对 MMA 的作用就像 TMA 对复制所做的那样，使其成为不消耗寄存器的单线程异步操作。因此，寄存器主要可用于其他任务，例如调度和融合尾声操作。

TMEM 的大小为每个 SM 256KB，并以 512 列和 128 行的二维方式组织，或者**车道**，32 位单元。这种固有的 2-D 结构也反映在 32 位地址中，其中位 31-16 表示通道 ID，而 15-0 表示列。这张图片来自于[PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-addressing)显示布局：

![](../images/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/tensor-memory-layout-9dcc40873b.png)

TMEM 使用动态分配`tcgen05.alloc`操作说明。此外，分配是以列为单位的，因此特别是在分配列时分配列的每个通道。分配的列数必须是 2 的幂且至少为 32。最后，必须使用以下命令显式释放 TMEM`tcgen05.dealloc`。 两个都`tcgen05.alloc`和`tcgen05.dealloc`必须从单个 warp 调用，并且同一个 warp 应该同时分配和释放。

请注意，`tcgen05.alloc`指令将分配的基址 32 位地址存储到共享内存中的给定位置。然后，TMEM 基地址应设置为 UMMA 累加器张量的偏移量，如下所示。

通常，数据获取*进入*TMEM 通过 UMMA 操作，并显式移动*出去*到寄存器使用`tcgen05.ld`用于后处理。线程还可以手动将数据加载到 TMEM 中，无论是从 SMEM 到`tcgen05.cp`或从寄存器通过`tcgen05.st`。然而，显式加载和存储的 TMEM 访问模式非常有限。 warpgroup 中的每个 warp 只能访问 32 个通道（warp 0 与通道 0-31 关联，warp 1 与通道 32-63 关联，依此类推）。此外，UMMA 操作和数据移动操作都需要特定的数据布局。对我们来说幸运的是，CUTLASS 提供了实用函数，我们稍后将介绍这些函数，这些函数简化了通过 swizzling 组织数据的过程。也就是说，有兴趣的人可以在中找到布局信息[PTX指南](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-layout-swizzling).

最后，除了UMMA操作和这些数据移动指令之外，没有其他操作从TMEM访问数据。换句话说，所有的预处理都必须发生*前*数据加载到TMEM，并且所有后处理都必须发生*后*从 TMEM 中检索数据。

## `tcgen05.mma`

尽管我们主要使用 CUTLASS 接口来执行此操作，但 PTX 文档是了解其功能的最佳来源。忽略一些可选参数，[PTX 语法](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-mma)为`tcgen05`MMA 操作采用以下形式之一：

```

tcgen05.mma.cta_group.kind   [d-tmem],  a-desc,  b-desc, idesc, enable-input-d;
tcgen05.mma.cta_group.kind   [d-tmem], [a-tmem], b-desc, idesc, enable-input-d;
.kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4 }
.cta_group = { .cta_group::1, .cta_group::2 }
```

在此示例中，我们将查看具有 FP32 累积 (.kind::f16) 的密集 FP16 GEMM。我们现在只考虑 1-CTA 案例 - 本系列的下一篇文章将讨论 2-CTA 版本。从[支持的矩阵形状表](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-kind-shapes)，我们看到 MMA 指令的形状为 64 x N x 16（N 为 8 的倍数）和 128 x N x 16（N 为 16 的倍数），这两种情况下 N 最多为 256。（对于所有数据类型，密集 GEMM 的 K 预计为 32 字节宽。）请注意，最大的 UMMA 原子为 128 x 256 x 16，是最大 WGMMA 原子的两倍。它的累加器恰好占据了 TMEM 的一半，这意味着可以在不牺牲性能的情况下对多个 UMMA 原子进行流水线处理。

操作数`a-desc`和`b-desc`是[共享内存描述符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#shared-memory-descriptor)，它们非常类似于[用于 WGMMA 的](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)。简而言之，这些是 64 位值，包含有关存储在 SMEM 中的矩阵的地址、布局和混合模式的信息。 （如果 A 源自 TMEM，则其描述符将被其 TMEM 地址替换。）SMEM 中的矩阵tile预计为 K 大调，尽管 MMA 指令能够转置它们，并且允许具有以下之一[一些预定义的混合模式](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-layout-swizzling)与用于 WGMMA 的类似。

除了矩阵描述符之外，`tcgen05.mma`还期望一个**指令描述符**（论点`idesc`）。这是一段 32 位元数据，包含数据类型和稀疏性信息等；完整的详细信息可以找到[这里](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-descriptor)。值得注意的是，指令描述符中的两位告诉指令转置 and/or 否定 A 和/或 B。此外，参数`enable-input-d`在执行 MMA 之前将累加器清零（操作 D = A * B）和保留累加器（D = A * B + D）之间切换。

累加器位于[透明的行主格式](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-path-layout-organization)在TMEM。由于寄存器中不保存任何数据，因此我们不再需要担心 WMMA 和 WGMMA 所需的复杂线程值布局。然而，请记住，在存储或后处理之前，数据必须复制到寄存器中，并且每个 warp 只能访问 TMEM 的 1/4。这意味着尾声需要整个warpgroup。

由于所有数据均用于`tcgen05.mma`位于 CTA 共享内存空间（TMEM 或 SMEM）中，该操作可以而且必须由 CTA 中的单个线程发出。

## `tcgen05.ld`

内存移动指令分为三种类型`tcgen05`: `ld`, `st`， 和`cp`。对于我们的讨论，我们将重点关注`ld`，用于将数据从TMEM复制到RMEM。基本版本是[PTX 指令`tcgen05.ld`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-ld)如下：

```

tcgen05.ld.sync.aligned.shape.num.b32    r, [taddr];

.shape = { .16x64b, .16x128b, .16x256b, .32x32b }
.num    = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 }
```

如图所示`.sync.aligned`预选赛，`tcgen05.ld`是一个 warp 范围的指令，其中 warp 中的所有线程必须执行相同的指令并作为 warp 同步，类似于之前的`ldmatrix`操作说明。

`tcgen05.ld`支持各种数据移动形状，如[PTX指南](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-shape)。它们一般用{lanes}x{bits}表示；我们的示例使用 32x32b，它对应于 32 位的 32 个通道（或数据路径），跨单个扭曲。下一个组件，`.num`，描述了在列维度中重复的次数。对于我们的示例，我们使用执行单次加载的 .x1。在单条指令中，一个 warp 最多可以加载lane * bits * num <= 128 kb (16 kB)，相当于每个线程128个32位寄存器。最后，回想一下，每个 warp 只能访问 128 个 TMEM 通道中的 32 个。

这张图来自于[PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-fragment-3232b)显示我们的`tcgen05.ld.sync.aligned.32x32b.x1.b32`手术：

![](../images/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/tcgen05-mma-fragment-3232b-151b9f379e.png)

我们可以看到每个线程从一个通道加载 32 位（或一列）并将其存储在寄存器中。该图还显示`.num = .x2`在这种情况下，第二次重复加载并为每个线程使用第二个寄存器。

我们的指导的论据只是`r`和`taddr`， 在哪里`r`是目标寄存器并且`taddr`是 TMEM 地址 — 请注意，这是*根据*TMEM 加载的tile的地址，并且在经线中的所有线程中都是相同的。

由于有大量的选择，自然的问题是如何选择正确的变体。通道数受所使用的 MMA 指令影响最大；不同的`tcgen05.mma`变体[导致不同的输出布局](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-organization)，以及不同的`tcgen05.ld`形状适合不同的情况。对于位宽和`.num`，考虑更多的是性能和资源。较大的重复将减少发出的指令数量，并且可以促进矢量化。然而，较大的`.num`价值观也[需要更多寄存器](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-num-shapes-ld)。所以这个值是一个调整参数。

现在我们了解了 UMMA 指令的功能，接下来我们讨论将用来访问它的 ZXQPH0ZXQ/ZXQPH1ZXQ 接口。与之前的 CUTLASS MMA 抽象一样，这是通过以下方式描述的：

- cute/arch/目录中的MMA_Atom，它是相应PTX指令的包装器；
- cute/atom/ 目录中的 MMA_Traits，其中包含 CuTe 布局和用于以 CUTLASS 原生方式与原子交互的其他元数据。

我们的[WGMMA教程](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)对此设计进行了更深入的解释。下面是[SM100_MMA_F16BF16_SS 的模板签名](https://github.com/NVIDIA/cutlass/blob/331a1f5b3fa3b6a9d9ef57c393d8719fb5510a32/include/cute/atom/mma_traits_sm100.hpp#L1090)，这是第一个 CuTe Blackwell 代码示例中使用的原子。

```

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
                                M, N, a_major, b_major,
                                a_neg, b_neg>>
{
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && 
                          cute::sizeof_bits_v<b_type> == 16, 
                          "SM100_MMA_F16BF16_SS supports 16bit types");
  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;
  // Logical shape-K is always 256bits, transform to units of elements
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;
  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();
  // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
...
}
```

这里的很多信息都透明地映射到我们已经看到的概念上`tcgen05.mma`指令：A 和 B 的 SMEM 描述符、D 的 TMEM 布局以及指令描述符。原子大小由模板参数 M 和 N 提供（如注释所示，K 由数据类型的位大小确定）。模板参数`a_major`, `b_major`, `a_neg`， 和`b_neg`用于填充指令描述符的转置位和取反位。这`accumulate_`成员（预计是`UMMA::ScaleOut::One`或者`UMMA::ScaleOut::Zero`) 提供 PTX 参数`enable_input_d`.

但布局中发生了一些有趣的事情。此前，`ThrID`布局用于将在 MMA 指令上协作的线程的逻辑索引映射到其物理线程 ID。对于warp宽度 MMA，`ThrID`曾是`Layout<_32>`，对于 WGMMA，它是`Layout<_128>`。这里它被简化为`Layout<_1>`。类似地，A、B 和 C 的 TV 布局在线程模式下的大小为 1。我们可能会认为这是因为该指令是单线程的，但事实远不止于此。*由于该指令是单线程的，因此所有线程布局都已重新调整用途，作为在 MMA 指令上协作的 CTA 的布局。*

目前，我们只会为每个 MMA 使用 1 个 CTA，这将导致相当数量的明显无关的 static-1，当我们在下一篇博客文章中逐步使用 2 个 CTA 时，其用途将变得更加清晰。

最后，简要说明原子名称的解剖结构。`SM100_MMA_F16BF16_SS`可以解构为以下几个部分。

- `SM100_MMA`: 指定指令。简单地说，UMMA 指令`sm100`.
- `F16BF16`：指定 A 和 B 可接受的输入类型。在这种情况下，可以是`fp16`或者`bf16`。请注意，这映射到`.kind`预选赛`tcgen05.mma`(例如，`.kind::f16`），而确切的输入类型由[指令描述符](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instuction-desc-kind-tf32-f16-f8f6f4).
- `SS`：指定A和B的内存位置。`SS`两者都在 SMEM 中，而`TS`A 位于 TMEM 中，B 位于 SMEM 中。
- 后缀：对于更复杂的情况还有其他后缀，例如块缩放或 2-SM UMMA。

请注意，与 Hopper 原子不同，MMA 的大小以及操作数或所使用的累加器的数据类型不是原子名称的一部分。相反，它们是由模板参数确定的。

接下来，我们来讨论一下中提出的实现[第一个 Blackwell CuTe 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu)。为了使讨论集中在 Blackwell 上，我们假设您对 CUTLASS GEMM 内核的典型格式有一定程度的熟悉。有关更多介绍性讨论，请参阅我们的[早期的博客系列](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/).

为了清楚起见，我们将我们的讨论大致分为五个部分：

1. GMEM 平铺机和切片机
2. SMEM 布局和混合
3. 输入和输出描述符
4. 同步和GEMM
5. 复制出TMEM

所有五个部分中反复出现的主要主题是分区发生在 CTA 之间，而不是线程之间。

## GMEM 平铺机和切片机

首先，我们需要将全局输入张量划分为tile，并将它们分配给 CTA 进行处理。对于此示例，我们没有多个 SM 在同一个 UMMA 上进行协作，因此此处 CTA 的平铺与 UMMA 的平铺相同（在重要设置中并非如此，我们将在以后的文章中看到）。所以我们创建了`tiled_mma`首先对象，然后根据选择平铺器的尺寸`tiled_mma`.

```

TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC,                 
                                                         128, 256,
                                                         UMMA::Major::K, 
                                                         UMMA::Major::K>{});
auto bM = tile_size<0>(tiled_mma);             // 1 MMA per CTA tile
auto bN = tile_size<1>(tiled_mma);             // 1 MMA per CTA tile
auto bK = tile_size<2>(tiled_mma) * Int<4>{};  // 4 MMA per CTA tile
auto mma_tiler = make_shape(bM, bN, bK);       // (MMA_M, MMA_N, MMA_K)
```

这里需要注意的一个区别是 4 的因数`MMA_K`是每 K 个tile的 MMA 数量，而不是 K 个tile的数量。这意味着每个 GMEM 到 SMEM 副本有 4 个 UMMA 调用，因为此副本是按tile完成的。

打印 MMA 显示以下内容。

```

TiledMMA
  ThrLayoutVMNK:  (_1,_1,_1,_1):(_0,_0,_0,_0)
  PermutationMNK: (_,_,_)
MMA_Atom
  ThrID:  	_1:_0
  Shape_MNK:  (_128,_256,_16)                  	// MmaM, MmaN, MmaK instruction size
  LayoutA_TV: (_1,(_128,_16)):(_0,(_1,_128))   	// TV -> MmaCoordinate mapping for A
  LayoutB_TV: (_1,(_256,_16)):(_0,(_1,_256))   	// TV -> MmaCoordinate mapping for B
  LayoutC_TV: (_1,(_128,_256)):(_0,(_1,_128))  	// TV -> MmaCoordinate mapping for C
```

正如我们在 MMA 原子中看到的那样，所有“线程布局”都被重新调整用途，以指在 MMA 上协作的 CTA 布局。在此示例中，每个仅使用一个 CTA`TiledMMA`，因此所有这些布局的大小均为 1。A、B 和 C 的值布局按预期显示其形状。然后我们得到每个 CTA 的以下 GMEM 张量：

```

print(gA);   // (_128,_64,4):(256,_1,_64)
print(gB);   // (_256,_64,4):(256,_1,_64)
print(gC);   // (_128,_256):(1024,_1)
print(gD);   // (_128,_256):(1024,_1)
```

我们看到静态整数`bM, bN, bK = _128, _256, _64`出现为这些布局的模式，以及动态整数 4，因为我们采用`K = 256`在这个例子中。

作为“线程布局”重新调整为“对等 CTA 布局”的另一个结果，`tiled_mma`现在由 CTA 对等点 ID 而不是线程 ID 进行切片。然而，在这个例子中，我们只有一个 CTA，所以我们可以简单地按`_0{}`.

```

ThrMMA cta_mma = tiled_mma.get_slice(_0{});   
Tensor tCgA = cta_mma.partition_A(gA);        // (MmaA, NumMma_M, NumMma_K, Tiles_K)
Tensor tCgB = cta_mma.partition_B(gB);        // (MmaB, NumMma_N, NumMma_K, Tiles_K)
Tensor tCgC = cta_mma.partition_C(gC);        // (MmaC, NumMma_M, NumMma_N)
Tensor tCgD = cta_mma.partition_C(gD);        // (MmaC, NumMma_M, NumMma_N)

print(tCgA); // ((_128,_16),_1,_4,4):((256,_1),_0,_16,_64)
print(tCgB); // ((_256,_16),_1,_4,4):((256,_1),_0,_16,_64)
print(tCgC); // ((_128,_256),_1,_1):((1024,_1),_0,_0)
print(tCgD); // ((_128,_256),_1,_1):((1024,_1),_0,_0)
```

此更改反映在切片 MMA 的名称中。在针对 Hopper 的 CuTe 示例中，切片 MMA 通常被标记为`thr_mma`，但现在它被称为`cta_mma`。最后，分区的 GMEM 张量具有从 MMA 原子大小 128x256x16 推导出来的预期布局。

### 处理集群

到目前为止，我们只讨论了 UMMA 的 1 SM 情况。然而，在每个 UMMA 涉及 2 个 SM 的情况下，UMMA 形状与 CTA 形状不同，我们需要对`tiled_mma`与对等点 CTA ID（i.e。CTA 在其对中的位置，0 或 1）。我们简要地离题以展示如何适应这种情况，并将更广泛的讨论推迟到本系列的第 2 部分。

每个 CTA 对必然由簇中一对相邻的 CTA 组成。这意味着我们可以按如下方式提取对等 CTA ID。

```

Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                         make_tile(typename TiledMMA::AtomThrID{}));
auto mma_coord_vmnk = make_coord(
                   blockIdx.x % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                   blockIdx.x / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
                   blockIdx.y,                                //    MMA-N coordinate
                   _);                                        //    MMA-K coordinate

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);   // Use Peer CTA coordinate
```

`cluster_layout_vmnk`用于创建识别 CTA 对的 cluster_shape ；`AtomThrID`是 1 或 2，具体取决于指定的 UMMA 原子是否使用 CTA 对。然后用它来计算 CTA 的 4 维坐标，其中第 0 个模式是对等 CTA ID。请注意，在这种情况下`size<0>(cluster_layout_vmnk)`为 1（没有 CTA 对），坐标简化为更熟悉的`(1, blockIdx.x, blockIdx.y, _)`.

最后，我们可以使用第0模式来切片`tiled_mma`。再次，对于这个特定的例子，`mma_v`始终为 0，因为只有 1 个 CTA。但在后面的例子中，`mma_v`将为 0 或 1。

## SMEM 布局和混合

我们现在有了全局张量的平铺器，因此我们有了副本的源端。接下来是目的地：SMEM。对于 A，目标张量，`tCsA`，应按形状组织`(MmaA, NumMma_M, NumMma_K) = ((_128,_16),_1,_4)`与GMEM布局保持一致。 CUTLASS 具有用于创建所需形状的实用函数。

```

auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler),
                                                           size<2>(mma_tiler)));
auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), 
                                                           size<2>(mma_tiler)));
```

为了优化 SMEM 访问，还应该对布局进行 swizzle，具体操作如下。

```

// Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{},
                                         mma_shape_A);
// Sw<3,4,3> o smem_ptr[16b](unset) o ((_256,_16),_1,_4):((_64,_1),_0,_16)
auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, 
                                         mma_shape_B);
```

这里，`Layout_K_SW128_Atom<TypeA>`是 K 大调 A 数据的 128 字节宽 swizzle`TypeA`。 swizzle 的宽度由连续维度中tile的大小决定。在这种情况下，K维度有4个大小为16的tile，半精度为2字节，因此宽度为`16*4*2=128`字节。有关 MMA 调配的更多详细信息，请参阅[这篇文章](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/).

与其他 CUTLASS 代码一样，此示例动态分配 SMEM 并将其作为`SharedStorage`结构。在这种情况下，`SharedStorage`保存 A 和 B 的tile，以及用于管理 MMA 异步的 mbarrier 对象。为了处理 TMEM 分配，`SharedStorage`还保存 TMEM 基地址的 32 位地址。

```

template <class TypeA,       	// Tensor A data type
          class TypeB,       	// Tensor B data type
          class ASmemLayout, 	// (MmaA, NumMma_M, NumMma_K, ...)
          class BSmemLayout> 	// (MmaB, NumMma_N, NumMma_K, ...)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t mma_barrier;  // Barrier to track MMA computation on SMEM
  alignas(16) cute::uint32_t tmem_base_ptr;  // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};
```

此示例使用自动矢量化从 GMEM 复制到 SMEM`cute::cooperative_copy`。我们也可以像往常一样编写 TiledCopy 或使用 TMA。

## 输入和输出描述符

UMMA 可以接受来自 SMEM 或 TMEM 的第一个输入，第二个输入必须在 SMEM 中，累加器必须在 TMEM 中。示例中使用的特定原子变体采用来自 SMEM 的两个输入。

为了创建描述符，我们使用`make_fragment`的方法`cta_mma`就像 Hopper 和早期的 GEMM 一样。

```

// Represent the SMEM buffers for A and B
Tensor tCsA = shared_storage.tensor_sA();      // (MmaA, NumMma_M, NumMma_K)
Tensor tCsB = shared_storage.tensor_sB();      // (MmaB, NumMma_M, NumMma_K)

Tensor tCrA = cta_mma.make_fragment_A(tCsA);
Tensor tCrB = cta_mma.make_fragment_B(tCsB);

Tensor tCtAcc = cta_mma.make_fragment_C(tCgC); // (MmaC, NumMma_M, NumMma_N)
```

就像 Hopper 的 WGMMA 中一样，操作数 Tensors 不是寄存器支持数据的张量，而是[SMEM 矩阵描述符](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#shared-memory-descriptor)。例如，打印 tCrA 显示

```

tCrA:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
```

每个 MMA 原子有一个描述符，平铺为`(NumMma_M, NumMma_K) = (_1, _4)`。我们之前已经介绍过矩阵描述符作为[我们的博客 WGMMA](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/).

这里的累加器张量是一个普通的 TMEM 支持的张量，但它的布局一开始可能很难掌握：

```

tCtAcc: tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
```

TMEM地址的步长为65536；这是因为我们之前讨论过的 TMEM 的 32 位寻址方案。该地址的前 16 位表示通道，后 16 位表示列。这里的技巧是`65536 = 1<<16.`例如坐标`(1,1)`变成：

`(1,1) = (1*1<<16) + 1 = x0001.0001`

这是对应于第 1 列的通道 1 的 32 位地址（十六进制）。

## GEMM 与同步

就像 Hopper 的 WGMMA 一样，UMMA 是异步的，因此需要同步。该示例为此使用了一些 CUTLASS 快捷方式和 mbarrier 周围的抽象。以下是显示工作流程的示例的摘录。

```

if (elect_one_warp && elect_one_thr) {
  cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ 1);
}
int mma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
__syncthreads();                

// Initial MMA overwrites the accumulators
tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
{
  // ... copy data in ...

  // Only one warp starts UMMAs
  if (elect_one_warp) {
    // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
      // Non-initial MMAs accumulate into the accumulators
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
    // Ensure MMAs are completed, only then we can reuse the A and B SMEM.
    cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
  }
  // All warps wait for MMAs to complete to avoid overwriting the A and B SMEM.
  cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
  mma_barrier_phase_bit ^= 1;
}

// ... copy data out ...
```

同步结构本质上与 TMA 所使用的相同。如果您想要 TMA 和同步的入门教程，请参阅我们之前的[博客](https://research.colfax-intl.com/tutorial-hopper-tma/)。值得注意的一件事是 mbarrier 是由属于将启动 UMMA 的 warp 的一个线程初始化的。

这`gemm`从 Hopper 示例中也应该熟悉调用和循环结构。需要注意的主要区别是只有一个经线发射 UMMA。记得只有一个*线*应该发出 PTX UMMA 指令。 CUTLASS 在 UMMA 原子的实现中在幕后选择该线程，因此实际上，调用`cute::gemm`来自单个线程会导致死锁。

最后值得一提的是`UMMA::ScaleOut::Zero`。这指示 UMMA 覆盖 TMEM，而不是累加预先存在的值。第一个之后`k_block`迭代，这被设置为`UMMA::ScaleOut::One`从而将结果累积起来。

## 复制出TMEM

一旦所有 MMA 完成，我们需要将累加器结果从 TMEM 复制到寄存器。这是使用 PTX 完成的`tcgen05.ld`操作说明。 CUTLASS 摘要`tcgen05.ld`作为复制原子，我们之前看到的不同变体表示为在复制原子中定义的不同复制特征[cute/atom/copy_traits_sm100.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm100.hpp)。我们的示例使用`SM100_TMEM_LOAD_32dp32b1x`原子。我们可以看到这如何转化为原子周围 PTX 包装器中的正确变体，可以在[cute/arch/copy_sm100.hpp](https://github.com/NVIDIA/cutlass/blob/331a1f5b3fa3b6a9d9ef57c393d8719fb5510a32/include/cute/arch/copy_sm100.hpp#L3333).

```

// 32 data path lanes, 32-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_32dp32b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    "{%0},"
                    "[%1];\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};
```

使用这个原子，我们可以设置一个 TiledCopy 将累加器结果从 TMEM 提取到 RMEM。请注意，与我们在本示例中看到的其余 CTA 级操作不同，我们回到了扭曲和线程级操作 - 因为数据必须移动到寄存器才能执行尾声。

```

// Create the tiled copy operation for the accumulator (TMEM -> RMEM)
TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
ThrCopy   thr_t2r_copy   = tiled_t2r_copy.get_slice(threadIdx.x);

//...

Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);    
Tensor tDgD   = thr_t2r_copy.partition_D(tCgD);     
using AccType = typename decltype(tCtAcc)::value_type;
Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));   
// Load TMEM -> RMEM
copy(tiled_t2r_copy, tDtAcc, tDrAcc);
```

这里我们使用一个专门的函数，`make_tmem_copy`，从复制原子和 TMEM 张量推导出 TV 布局并创建 TiledCopy。关于这个函数需要了解的一件重要的事情是*它被硬编码为使用 4 个扭曲或 1 个warpgroup。*如前一节所述，TMEM 的某些区域只能由基于扭曲索引 mod 4 的warpgroup中的相应扭曲访问。[PTX 手册中的图表](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#layout-d-m-128-cta-group-1)在我们的例子中,显示了如何将数据分配给扭曲:：

![](../images/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/tcgen05-data-path-layout-d1-1f136eac6b.png)

这[下图来自PTX手册](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-d2)显示该映射到的 TMEM 地址。

![](../images/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/tcgen05-data-path-layout-d2-ec2c03f5c8.png)

要了解 CuTe 如何处理此副本，我们可以转向特征结构，可在[cute/atom/copy_traits_sm100.hpp](https://github.com/NVIDIA/cutlass/blob/b84e9802d84b16bcb4e92338fcf0a04785df9236/include/cute/atom/copy_traits_sm100.hpp#L2110).

```

template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b1x>
     : TMEM_LOAD_Unpack<SM100_TMEM_LOAD_32dp32b1x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_32, _32>, Stride< _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32, _1024>, Stride< _0, _1>>;
  using DstLayout = Layout<Shape <_32, _32>, Stride<_32, _1>>;
  using RefLayout = SrcLayout;
};
```

布局`ThrID`定义从逻辑线程 ID 到线程束中线程索引的映射；价值`32`意味着这是一个扭曲范围的操作。

`ValID`告诉我们从逻辑位id到位地址的映射；例如，位 35 被映射为`ValID`布局到车道 1 上的第 3 位。布局具有形状（位，车道）和车道的步幅，`TMEM::DP_b`， 是`1<<21`; `1<<16`正如我们之前看到的，来自 TMEM 寻址方案，额外的 5 来自单元格`1<<5=32`位宽。

`SrcLayout`为我们提供了从（src-thread，src-bit）到位的映射。此加载是一个 warp 范围内的操作，并且输入基 src 地址在整个 warp 上是相同的。因此线程值被抑制（步长为 0），以便将 src 位映射到位。

最后，`DstLayout`显示（dst-thread，dst-bit）到位的映射。布局的形状`<32,32>`告诉我们每个线程负责写出 32 位（1 个寄存器）。请注意，此布局对于`32dp32b`因为 TMEM 中的泳道和列直接转换为输出中的行和列。但对于更复杂的负载模式，我们需要此布局来确定输出 RMEM 位如何映射到逻辑位索引。

现在返回到代码，从此原子创建的 TiledCopy 用于对输出矩阵进行分区。然后由线程 ID 对分区进行切片以获得每线程张量。给定我们的 MMA 大小为 128×256，我们得到为线程 0 打印的以下张量（显示`tCtAcc`再次方便参考）：

```

// reproduced from above
tCtAcc: tmem_[32b](0x0000.0000) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
// new tensors for tmem -> rmem copy
tDtAcc: tmem_[32b](0x0000.0000) o ((_32,_1),_256,_1,_1):((_65536,_0),_1,_0,_0)
tDrAcc: ptr[32b](0x705671fff290) o ((_1,_1),_256,_1,_1):((_0,_0),_1,_0,_0)
```

我们可以看到128×256的MMA尺寸直接体现在`tCtAcc`。隔断`tDtAcc`是一个*每线程*映射到 TMEM 地址的张量。再次注意，同一 warp 中的每个线程统一读取相同的 TMEM 地址，这解释了值模式的子布局 (_32, _1) : (_65536, _1)。 4 个warp上有 128 个线程，这涵盖了 M 模式。第一个模式表示重复 256 次以覆盖 N 模式，因此我们得到了 128×256 的tile。最后两个 1 值是 M-tiles 和 N-tiles，在我们的例子中是 1。上`tDrAcc`另一方面，主要区别在于它代表寄存器。因此，由于每个线程负责 TMEM 中的一个 32 位单元，因此我们只看到 (_1, _1) 作为值模式。再次，128 个线程跨 4 个warp，这涵盖了 M 模式。其他模式与此相同`tDtAcc`.

最后，一旦累加器被复制到 RMEM，就可以对其进行后处理（e.g.`axpby`)，然后再存储回 GMEM。

对于基本示例，还有一个附加主题需要讨论：TMEM 分配和释放。我们可以使用 CuTe 帮助程序类来完成此操作[cute::TMEM::Allocator1Sm](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/tmem_allocator_sm100.hpp)，它提供了一个接口`tcgen05.alloc`和`tcgen05.dealloc`上面讨论的功能。基本模式如下。

```

// instantiate the allocator 
cute::TMEM::Allocator1Sm tmem_allocator{};

if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
}
__syncthreads();
tCtAcc.data() = shared_storage.tmem_base_ptr;   // move accumulator offset

// rest of kernel

if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
```

正如前面部分所讨论的，一个 warp 执行分配，传递多个列和一个指向共享内存中 32 位值的指针；这`allocate`方法然后存储分配的 TMEM 的起始（最低（通道、列））的 32 位地址。尽管此 MMA 指令仅需要 256 列，但为了简单起见，内核分配了 TMEM 的所有 512 列。请注意，虽然只有一个线程将 TMEM 地址传递给 MMA 指令，但所有线程都需要它从 TMEM 加载数据作为尾声，要求它通过共享内存传递。最后，同样的扭曲被称为`allocate`还得打电话`free`。作为一个稍微高级的功能，`release_allocation_lock`方法是一个包装器[`tcgen05.relinquish_alloc_permit`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit);显然，这是保证 CTA 不会执行任何进一步的 TMEM 分配，从而允许未来的 CTA 排队等待相同的 SM。您可以在中查看一些更完整的 TMEM 管理示例[CUTLASS sm100 GEMM 内核](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L535).

为了帮助 TMEM 管理，nvcc 添加了标志`--g-tensor-memory-access-check`。启用此标志后，在运行时内核将在任何未初始化或越界 TMEM 访问上出错并打印错误消息。

在这篇文章中，我们讨论了 Nvidia Blackwell GPU 上可用的新功能，然后通过回顾如何使用这些功能[第一个 CuTe Blackwell 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu)。我们观察到，CUTLASS GEMM 内核的主要概念和整体结构与 Blackwell 架构没有变化。也就是说，我们在示例中观察到的两个主要变化是：

1. UMMA原子处于CTA级别而不是线程级别，因此围绕TiledMMA和同步模型的各种构造必须相应更新（例如，单线程异步发出UMMA）；
2. UMMA 累加到新的张量内存中，TMEM 必须手动管理，并且必须使用特殊的 TiledCopy 将累加器从 TMEM 复制到寄存器中。

我们在这篇文章中讨论的示例仅处理单个 SM UMMA 指令，并且仅使用了简单的簇形状`<1,1,1>.`然而，集群级协作是Blackwell内核的重要组成部分。在下一篇文章中，我们将讨论使用多播和 2SM UMMA 处理重要簇形状的示例。

Cris Cecka、Mihir Awatramani，“使用 CUTLASS 编程 Blackwell Tensor Cores”，GTC 2025，[https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/).
