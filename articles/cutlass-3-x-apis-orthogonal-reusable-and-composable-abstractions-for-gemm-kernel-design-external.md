---
title_zh: "CUTLASS 3.x：GEMM 内核设计的正交、可重用和可组合抽象"
title_en: "CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design"
source_url: "https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/"
published_at: "2025-07-16"
source_site: "NVIDIA Technical Blog"
external: true
english_markdown: "articles-en/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external.en.md"
---
# CUTLASS 3.x：GEMM 内核设计的正交、可重用和可组合抽象

原文标题：CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design

英文对照：[articles-en/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external.en.md](../articles-en/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external.en.md)

GPU 上的 GEMM 优化是一个模块化问题。高性能实现需要指定超参数，例如tile形状、数学和复制指令以及warp specialization方案。这些超参数在很大程度上是相互独立的；此外，最佳选择可能会因硬件、问题形状或其他用户需求而有很大差异。

通过 3.x 重新设计，CUTLASS 旨在通过可组合、正交构建块的分层系统最大限度地覆盖 GEMM 实现的空间，同时提高代码可读性并将支持扩展到后来的 NVIDIA 架构，例如 Hopper 和 Blackwell。由于这种设计理念与 GPU 的分层硬件设计相关，因此它也可以成为其他 GPU 应用的不错选择 - 例如，[FlashAttention-3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)在其设计中使用熟悉的 CUTLASS 抽象。

在 CUTLASS 博客系列的第二篇博文中，我们将探讨 CUTLASS 3.x 中 GEMM 分层系统背后的设计原理，并解开 CUTLASS 如何从 中介绍的低级 CuTe 抽象构建 GEMM 内核。[第 1 部分](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels).

## CUTLASS 3.x 中的新概念 GEMM 层次结构[https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#a_new_conceptual_gemm_hierarchy_in_cutlass_3x](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#a_new_conceptual_gemm_hierarchy_in_cutlass_3x)

CUTLASS 3.x 开发了[概念性 GEMM 层次结构](https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api_3x.md)这与特定的硬件功能无关。它的结构分为五层：

![图 1. 独立于硬件的 CUTLASS GEMM 层次结构的概念图](../images/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external/CUTLASS-GEMM-hierarchy-png-4f1eb1ab5f.webp)

- **原子层**：特定于架构的指令以及相关的元信息`cute::Mma_Atom<>`和`cute::Copy_Atom<>`
- **平铺 MMA/Copy**：空间微内核，允许架构特定原子的任意交错和平铺`cute::TiledMma<>`和`cute::TiledCopy<>`
- **集体层**：时间微内核，使用特定于体系结构的同步来协调一个或多个空间微内核的执行，以计算单个输出 tile`cutlass::gemm::collective::CollectiveMma<>`, `cutlass::epilogue::collective::CollectiveEpilogue<>`
- **内核层**：用于在 threadblocks/clusters 网格上执行内核的设备代码`cutlass::gemm::kernel::GemmUniversal<>`
- **设备层**：主机端设置和接口`cutlass::gemm::device::GemmUniversalAdapter<>`

每一层都充当前一层抽象的组合点，可以使用模板参数进行高度自定义。用户可以坚持使用最高层，相信 CUTLASS 的编译时逻辑可以提供高性能的 GEMM 实现，也可以选择使用较低级别的层次结构公开的高级修改。Atom 和 Tiled MMA/Copy 层提供的空间微内核是 CuTe 的领域，并在第 1 部分中进行了讨论。本文的其余部分将介绍在更高层中提供的 GEMM 的时间和内核级组织。

以下是如何在 CUTLASS 3.x 中定义 GEMM 内核的基本示例：

```

// Step 1: Generate the required collective layer mainloop specialization
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Step 2: Specify the collective layer epilogue type
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;

// Step 3: Compose the mainloop and epilogue together at the kernel layer
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 4: Wrap up the kernel::GemmUniversal kernel class
// with the device adapter to obtain a host-side handle to the kernel
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

## 集体层：Mainloop[https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#collective_layer_mainloop](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#collective_layer_mainloop)

一个**集体**是一组相互协作执行工作的线程，并且可以并行重复以形成整个内核。一般来说，这是一个线程块或集群。TiledMMA 和 TiledCopy 对象描述并行工作器的空间分配以计算和复制工作（e.g、扭曲、warpgroup，甚至 Blackwell MMA 的线程块），而 Collective 层则负责通过设置管道和warp specialization方案以及使用硬件加速同步原语来管理管道和异步操作来临时组织这项工作。
CUTLASS 3.x GEMM 内核包含**集体主循环**，一个实例[GEMM](https://github.com/NVIDIA/cutlass/blob/b78588d1630aa6643bf021613717bafb705df4ef/include/cutlass/gemm/collective/collective_mma_decl.hpp)类模板定义了由单个集合执行的单个主循环迭代的基本成分，最重要的是加载和 MMA 过程。集体主循环可以这样定义：

```

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
  DispatchPolicy,
  TileShape,
  ElementA, // dtype, e.g. float
  StrideA,  // e.g. Stride<_1, int> for M-major
  ElementB, StrideB,
  TiledMma,
  GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,
  GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB
>;
```

集体主循环是来自较低层的抽象的组合点：TiledMma、每个操作数的 GMEM 到 SMEM 加载的 TiledCopy 以及用于寄存器源 MMA 的 SMEM 到 RMEM 加载的可选复制原子。这些抽象很大程度上是正交的，允许不同的 MMA 操作与不同的复制操作组合，同时最大限度地提高代码重用。

可以说最重要的部分是[**派遣政策**](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/gemm/dispatch_policy.hpp#L185)，它将主循环专门化定义为特定算法或 GPU 架构。例如，调度策略`MainloopSm90TmaGmmaWarpSpecialized`将 CollectiveMma 专门化为 Hopper TMA warp specialization实现。这本身就是一个模板，可以在管道阶段、集群形状和内核调度的选择上进行参数化，例如[乒乓球或合作社](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md#hopper-warp-specialization)对于 Hopper GEMM 内核。

专门的集体主循环实现的示例可以在[GEMM集体](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective)文件夹。

## 集体建设者[https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#the_collective_builder](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#the_collective_builder)

CollectiveMma 具有各种调谐旋钮，允许用户根据 TiledCopy 和 TiledMma 对象精确指定 GEMM 主循环，但这种灵活性也带来了复杂性。通常，用户希望从有关流水线、硬件功能和资源可用性的高阶考虑中推导出这些对象和关联的 SMEM 布局。CUTLASS 也可以使用以下方法执行此推导**集体建设者**界面。使用 CollectiveBuilder 的主循环声明如下所示：

```

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag,       // e.g. cute::arch::Sm90 for Hopper
  OpClass,       // e.g. cute::arch::OpClassTensorOp for Tensor Core
  ElementA, LayoutA, AlignmentA,
  ElementB, LayoutB, AlignmentB,
  ElementAccumulator,
  TileShape, ClusterShape,
  StageCount,    // e.g. cutlass::gemm::collective::StageCountAuto
  KernelSchedule // e.g. cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

模板参数从用户友好的标准中进行选择，并使用它们将较低级别的参数推导到 CollectiveMma 模板：

- **建筑学专业：**GPU 架构和 MMA 运算符的类型（例如，SIMT 或 Tensor Core）。
- **操作数和累加器信息：**操作数和累加器的数据类型，以及全局内存中操作数的对齐和编译时布局信息（例如，行优先或列优先）。
- **瓷砖形状：**用于推导TiledMma和TiledCopy对象以及SMEM布局。
- **日程安排信息：**调度算法使用集群形状、管道阶段计数和内核调度。阶段计数和内核调度参数有默认的“自动”选项，它告诉 CUTLASS 尝试自动为给定的架构和参数选择最佳的选项。

## 集体层：结语[https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#collective_layer_epilogue](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#collective_layer_epilogue)

这**集体尾声**是集体 API 的另一半。它在每次主循环迭代之后处理tile的后处理和输出存储的时间编排。与主循环一样，这意味着集体尾声是复制操作（输出存储）和一些数学运算（通常是元素运算，但也可能包括归约）的组合点。与主循环不同，这些数学运算本身可以通过**尾声访客树**（EVT）形式主义。这对于 AI 工作负载特别有用，这些工​​作负载经常需要在 GEMM 之后立即计算激活函数。 CUTLASS 的集​​体尾声处理**融合**将此激活函数放入内核中，消除了不必要的数据移动。

CUTLASS 有几个尾声，定义[这里](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective)在 GitHub 上。模板参数在实现之间差异很大，但通常包括以下信息：

- 有关矩阵 C 和 D 的数据类型和编译时布局信息。
- 指定任何附加后处理的融合操作。
- 适用于 GMEM 存储和任何 SMEM 暂存的 TiledCopy 操作。
- 调度策略，与集体主循环一样，包含有关集群大小、TMA 使用、warp specialization等信息。

这[尾声的 CollectiveBuilder](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/epilogue/collective/collective_builder.hpp)呈现更统一、更高级的接口：

```

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag,
  OpClass,
  TileShape,
  ClusterShape,
  EpilogueTileType,
  ElementAccumulator,
  ElementCompute,
  ElementC, GmemLayoutTagC, AlignmentC,
  ElementD, GmemLayoutTagD, AlignmentD,
  EpilogueScheduleType,
  FusionOpOrCallbacks
>::CollectiveOp;
```

其中许多参数在主循环构建器中是熟悉的，但也有一些是新的：

- 尾声可以将 CTA tile划分为更小的tile，以实现更好的数学复制重叠。
- 累加器（主循环的输出）现在是尾声的输入。尾声计算可以在不同的中间数据类型中进行（由下式给出）`ElementCompute`).
- CUTLASS 提供多种选择[常见的融合操作](https://github.com/NVIDIA/cutlass/blob/b84e9802d84b16bcb4e92338fcf0a04785df9236/include/cutlass/epilogue/fusion/operations.hpp)， 例如`D = activation(alpha * AB + beta * C)`。用户还可以使用以下命令构建定制的融合操作[尾声访客树](https://github.com/NVIDIA/cutlass/blob/b78588d1630aa6643bf021613717bafb705df4ef/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp)。有关尾声访问者树的更多信息，请参阅[这个科尔法克斯教程](https://research.colfax-intl.com/epilogue_visitor_tree/).
- [尾声时间表类型](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/epilogue/dispatch_policy.hpp)定义 TMA 和warp specialization的用法。默认`EpilogueScheduleAuto`告诉 CUTLASS 尝试推导出最佳选项。

要查看两个 Collective Builders 的实际操作，我们参考 CUTLASS[实施例49](https://github.com/NVIDIA/cutlass/blob/389e493055f981bfdc6d4348f823191ca7b9fddd/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu)对于 Hopper 和[实施例71](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu)对于 Blackwell。

## 内核层[https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#kernel_layer](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#kernel_layer)

集体层完全定义了集体在内核执行期间完成的计算。将集合扩展到覆盖整个动态大小的问题空间的线程块或集群网格上，这就是**内核层**。内核层通过将加载、存储、MMA 等原始程序拼接在一起，将集体主循环和集体尾声组装成设备内核。

内核层的入口点API是类[cutlass::gemm::kernel::GemmUniversal](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/gemm_universal_decl.h)，这是一个无状态通用设备内核，它将 GEMM 实现为集体主循环和集体尾声的组合。*无国籍*意味着调用者通过向内核传递参数来管理内核的状态。*普遍的*意味着`GemmUniversal`是 2.x 和 3.x GEMM 内核的入口点。对于 3.x API，基本用法`GemmUniversal`看起来像这样：

```

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, // e.g. Shape<int, int, int> for a fully generic GEMM
    CollectiveMainloop,
    CollectiveEpilogue
>;
```

与`TiledMma`和`TiledCopy`, `CollectiveMainloop`和`CollectiveEpilogue`是通过组成的正交抽象`GemmUniversal`。第一个模板参数（问题形状）主要用于在普通 GEMM（具有 3 级问题形状）和批处理 GEMM（具有 4 级问题形状）之间进行选择，但如果需要，也可以静态约束某些问题维度。

的实例化`GemmUniversal`在以下形式的文件中找到`cutlass/gemm/kernel/sm*_gemm_*.hpp`， 和`GemmUniversal`主要是根据调度`KernelSchedule`集体主循环的参数。所有实例化都呈现一致的接口：

- 用于将参数传递给内核的接口，包括问题形状、有关硬件的信息、张量的指针和布局以及尾声参数。
- 静态初始化函数，用于获取网格和块尺寸，检查内核是否可在硬件上实现，并为尾声或tile调度程序所需的任何归约操作或全局屏障设置全局内存工作空间。
- 最重要的是，它们将内核逻辑实现为`operator()`。这是一个*设备*函数——尽管内核层包含内核执行的所有逻辑，但它尚未公开从主机启动它的方法。

例如，定义了 Blackwell 的 TMA 扭曲专用内核[这里](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp).

## 平铺调度[https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#tile_scheduling](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#tile_scheduling)

内核层也是指定tile调度器的组成点。正如内核调度定义了集合内工作的时间组织一样，tile scheduler定义了集合之间工作的顺序和分配。对于最基本的tile scheduler，每个输出切片分配一个 CTA。 CUTLASS 3.x 为 Hopper 实现了两个额外的tile scheduler：**执着的**调度程序为每个 SM 启动一个 CTA 并具有每个 CTA (*潜在地*）在其生命周期内计算多个输出 tile——并且**Stream-K**调度程序，它也是持久的，但另外沿着 K 模式划分一些输出tile工作，以实现更好的负载平衡。在 Blackwell 架构上，人们使用调度器[集群启动控制](https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_cluster_launch_control.md)。有关切片调度的更深入信息，请参阅[这个科尔法克斯教程](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/).

我们可以扩展上面的内核以使用 Stream-K tile scheduler：

```

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler
>;
```

[CUTLASS 示例 74](https://github.com/NVIDIA/cutlass/blob/main/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu)是使用 Stream-K 调度程序的更详细示例。

## 设备层[https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#device_layer](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#device_layer)

用于内核启动的主机端逻辑，包括通过集群支持或在不同设备或 CUDA 流上启动，在**设备层**。设备层的主要入口点是`cutlass::gemm::device::GemmUniversalAdapter`，其中包含一个`GemmUniversal`内核位于有状态、可重用的句柄中。*有状态的*意味着句柄实例包含内核运行所需的状态（即，它自己管理内核参数）。*可重复使用的*意味着同一个句柄实例可以使用不同的参数多次调用内核。

`GemmUniversalAdapter`已实施[这里](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/gemm_universal_adapter.h)在 GitHub 上。这个例子展示了我们如何使用`GemmUniversalAdapter`启动内核：

```

using GemmHandle = cutlass::gemm::kernel::GemmUniversalAdapter<GemmKernel>;
using Arguments = typename GemmHandle::Arguments;    // surfaced from GemmKernel
Arguments args {
    cutlass::Gemm::kBatched,                   // mode (here batched GEMM)
    cute::make_shape(M, N, K, L),              // problem shape
    {A, stride_A, B, stride_B},                // mainloop args
    {{alpha, beta}, C, stride_C, D, stride_D}, // epilogue args
    make_kernel_hardware_info(device_id),      // hardware info
    {}                                         // scheduler args (here default)
};
GemmHandle gemm;

// Check that problem can run with given shape and hardware
cutlass::Status status;
status = GemmHandle::can_implement(args);
if (status != cutlass::Status::kSuccess) {
  std::cerr << "Problem not supported\n";
  exit(EXIT_FAILURE);
}

// Set up global memory workspace
size_t workspace_size = GemmHandle::get_workspace_size(args);
cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

// Initialize GEMM handle state from arguments
status = gemm.initialize(args, workspace.get());
if (status != cutlass::Status::kSuccess) {
  std::cerr << "Failed to initialize GEMM kernel\n";
  exit(EXIT_FAILURE);
}

// Launch kernel
status = gemm.run();  // can supply CUDA stream and CUDA host adaptor here
if (status != cutlass::Status::kSuccess) {
  std::cerr << "Failed to launch GEMM kernel\n";
  exit(EXIT_FAILURE);
}
```

## **概括**[https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#summary](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/#summary)

在这篇文章中，我们讨论了 CUTLASS 库在概念上如何组织为层次结构，其中每一层的对象都是根据较低层的正交对象组成的。该设计允许深度定制 GEMM 实现，并具有高水平的代码重用性。在本系列的下一篇也是最后一篇文章中，我们将介绍 CUTLASS 4.0 中引入的更改，特别是 CuTe Python DSL。

欲了解更多信息，您可以在以下网址下载该软件[GitHub](https://github.com/NVIDIA/cutlass)，阅读我们的[文档](https://docs.nvidia.com/cutlass/index.html)，或加入我们的[开发者论坛](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/cuda-programming-and-performance/7)进行更深入的讨论。

**致谢**

我们想感谢延长[CUTLASS团队](https://github.com/NVIDIA/cutlass/blob/main/CONTRIBUTORS.md)他们的集体努力使这项工作成为可能。
