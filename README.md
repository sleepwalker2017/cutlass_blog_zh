# CUTLASS / CUDA 博客中文导读

这个仓库整理了 Colfax Research 上与 CUTLASS / CUDA 相关的 12 篇博客，提供：

- 中文译文：[`articles/`](articles)
- 英文原文整理稿：[`articles-en/`](articles-en)
- 本地图片资源：[`images/`](images)
- 完整清单：[`index.md`](index.md)

中文稿适合顺序阅读，英文稿适合后续对照润色。代码块、图片和原始链接均已保留。

## 建议阅读路径

如果你想系统理解 CUTLASS、Hopper、Blackwell 以及 GEMM 内核设计，建议按下面的顺序读。

### 1. 先建立 GEMM 内核直觉

1. [CUTLASS 教程：在 NVIDIA® Hopper™ GPU 上使用 WGMMA 进行快速矩阵乘法](articles/cutlass-tutorial-wgmma-hopper.md)
   英文对照：[English](articles-en/cutlass-tutorial-wgmma-hopper.en.md)
   先理解 Hopper 上最核心的 WGMMA 原语、warpgroup、SMEM/RMEM 和 `cute::gemm` 的关系。
2. [CUTLASS 教程：使用流水线进行高效的 GEMM 内核设计](articles/cutlass-tutorial-design-of-a-gemm-kernel.md)
   英文对照：[English](articles-en/cutlass-tutorial-design-of-a-gemm-kernel.en.md)
   接着理解 pipeline、TMA、warp specialization、multistage，以及“怎么把 Tensor Core 喂满”。
3. [CUTLASS 教程：Persistent Kernels 与 Stream-K](articles/cutlass-tutorial-persistent-kernels-and-stream-k.md)
   英文对照：[English](articles-en/cutlass-tutorial-persistent-kernels-and-stream-k.en.md)
   最后把视角拉到网格级，理解 wave quantization、persistent kernel、tile scheduler、Stream-K。

这三篇是主干，读完之后对 CUTLASS GEMM 的整体结构会比较清楚。

### 2. 再补 Hopper / CuTe / 数据搬运细节

1. [CUTLASS 教程：掌握 NVIDIA® Tensor Memory Accelerator (TMA)](articles/tutorial-hopper-tma.md)
   英文对照：[English](articles-en/tutorial-hopper-tma.en.md)
   专门补齐 Hopper 上 TMA 的编程模型与同步方式。
2. [教程：CUTLASS 中的矩阵转置](articles/tutorial-matrix-transpose-in-cutlass.md)
   英文对照：[English](articles-en/tutorial-matrix-transpose-in-cutlass.en.md)
   适合拿来理解 layout、copy、tile 组织和数据重排。
3. [CUTLASS 中的尾声与尾声访问者树融合](articles/epilogue_visitor_tree.md)
   英文对照：[English](articles-en/epilogue_visitor_tree.en.md)
   适合在理解 mainloop 之后，再补 epilogue fusion / EVT。
4. [CUTLASS 3.x：GEMM 内核设计的正交、可重用和可组合抽象](articles/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external.md)
   英文对照：[English](articles-en/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external.en.md)
   用来从 API 设计层面回看 CUTLASS 3.x 的层次结构。

### 3. 最后进入 Blackwell 专题

1. [CUTLASS 教程：使用张量内存为 NVIDIA® Blackwell GPU 编写 GEMM 内核](articles/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus.md)
   英文对照：[English](articles-en/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus.en.md)
   先看 Blackwell 上 TMEM / UMMA 的基本编程方式。
2. [CUTLASS 教程：NVIDIA® Blackwell GPU 上具有线程块集群的 GEMM](articles/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus.md)
   英文对照：[English](articles-en/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus.en.md)
   再看 cluster、pair-UMMA、跨 CTA 协作与同步。
3. [CUTLASS 教程：NVIDIA® Blackwell GPU 上的子字节 GEMM](articles/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus.md)
   英文对照：[English](articles-en/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus.en.md)
   重点看 `f8f6f4`、sub-byte 数据布局与 TMA 解包。
4. [CUTLASS 教程：使用 NVIDIA Blackwell GPU 进行硬件支持的块扩展](articles/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus.md)
   英文对照：[English](articles-en/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus.en.md)
   最后读 block scaling，把低精度与缩放策略连起来。

### 4. 如果你关心 PyTorch 接入

- [教程：PyTorch 中 CUDA 库的 Python 绑定](articles/tutorial-python-binding-for-cuda-libraries-in-pytorch.md)
  英文对照：[English](articles-en/tutorial-python-binding-for-cuda-libraries-in-pytorch.en.md)
  这篇和 CUTLASS 核心设计关系没那么强，但很适合把自定义 CUDA / CUTLASS 库接进 PyTorch。

## 术语约定

为了保证技术可读性，仓库里的译文默认采用下面的风格：

- 稳定专有名词优先保留英文：`warp`、`warpgroup`、`Tensor Core`、`WGMMA`、`TMA`、`Stream-K`
- 常见缩写首次出现时给中文解释：例如 `GEMM`、`SM`
- 代码、API、类型名、数学符号不翻译
- 某些中文译法如果会降低可读性，会回退到英文术语，例如 `threadblock rasterization`

## 当前状态

- 12 篇中文稿已生成
- 12 篇英文对照稿已生成
- 58 张图片已本地化
- 已做一轮术语和可读性润色，但仍建议后续继续按主题做人工精修
