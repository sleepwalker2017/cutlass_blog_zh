---
title_zh: "CUTLASS 教程：掌握 NVIDIA® Tensor Memory Accelerator (TMA)"
title_en: "CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA)"
source_url: "https://research.colfax-intl.com/tutorial-hopper-tma/"
published_at: "2024-06-24"
source_site: "Colfax Research"
external: false
english_markdown: "articles-en/tutorial-hopper-tma.en.md"
---
# CUTLASS 教程：掌握 NVIDIA® Tensor Memory Accelerator (TMA)

原文标题：CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA)

英文对照：[articles-en/tutorial-hopper-tma.en.md](../articles-en/tutorial-hopper-tma.en.md)

TMA (Tensor Memory Accelerator) 是 NVIDIA Hopper™ 架构中引入的一项新功能，用于在 GPU 的全局内存 (GMEM) 与其线程块 (即， CTA) 的共享内存 (SMEM) 之间进行异步内存复制。与之前的方法相比，TMA 具有许多优点，例如（1）通过促进[warp专用](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization)内核通过异步进行调度，以及（2）通过 TMA 复制描述符以单线程方式处理辅助复制数据（例如地址和步幅）的计算，这既提高了寄存器效率，又必须处理预测（例如，越界检查）。这些优点在 ​​NVIDIA 中得到了很好的体现[技术博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)和[Hopper调音指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)，我们强烈推荐给读者，以帮助他们了解 TMA 设计背后的基本原理。

与这些来源相反，这篇博客文章的重点是实现对如何编写使用 TMA 的内核的操作理解。在整个过程中，我们依靠[CuTe库](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)，其中 TMA 通过封装较低级别 GPU 指令的 API 公开。这些指令包括PTX指令`cp.async.bulk.tensor`和[`cp.reduce.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)，以及[cu张量图](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)操作数，我们也将在本文中讨论。

我们将这篇博文分为三个主要部分：第一个关于 TMA 负载，第二个关于 TMA 存储，最后第三个涵盖更高级的操作，例如 TMA 存储归约和 TMA 负载多播。本质上，TMA 负载将数据从 GPU 的 GMEM 复制（“加载”）到其 CTA 的 SMEM 之一，而 TMA 存储将数据从 CTA 的 SMEM 复制（“存储”）到 GPU 的 GMEM。由于 TMA 加载、TMA 存储和更高级的变体共享许多概念，因此我们将在 TMA 加载部分介绍大量必要的概念，并在后续部分中仅关注剩余的差异。

另外，考虑到 TMA 是一个异步操作（在[异步代理](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#async-proxy)），我们需要使用某些内存一致性强制工具，例如异步内存屏障（即，[`mbarrier`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)) 和异步内存栅栏 (即，[`fence.proxy.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)），以确保内核的正确行为。同步本身就是一个广泛讨论的话题，因此我们只会在实际使用所需的范围内介绍这些概念。

最后，对于正在寻找涵盖许多相同点但未提及 CUTLASS 或 CuTe 概念的资源的读者，我们建议[TMA 的讲解](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access)在 CUDA® 编程指南中。

## TMA 负载

TMA 将数据从 GMEM 加载复制到 SMEM。在本节中，我们演示如何编写一个使用 TMA 负载来实现此目标的内核。使用 TMA load 的内核与使用其他内存复制方法的内核有很大不同，因此我们将首先展示如何为一个简单的示例任务编写这样的内核。然后，我们将解释其中涉及的概念。

#### **示例任务**

为了演示 TMA 负载的用法，我们考虑平铺 2D 行主矩阵的简单任务。我们给定一个矩阵`A`形状的`[m,n]`和两个正整数`CTA_M`和`CTA_N`。注意`CTA_M`和`CTA_N`在编译时已知，而`m`和`n`在运行时通过矩阵给我们`A`。为了简单起见，我们还假设`m % CTA_M == n % CTA_N == 0`，尽管我们稍后会看到这个要求可以放宽。

我们推出了具有尺寸的 CTA 网格`{m/CTA_M, n/CTA_N, 1}`，其中 SMEM`(i,j)`-th CTA 持有`(i,j)`第一个具有形状的tile`[CTA_M, CTA_N]`从`A`。我们可以在中描述这个任务`numpy`伪代码为：

```

A = np.random.uniform(M, N)
for i in range(M):
  for j in range(N):
    cta_i_j = A.reshape(M // CTA_M, CTA_M, N // CTA_N, N)[i, :, j, :]
```

**两步过程。**为了执行此任务，我们使用 TMA 负载。在CuTe中，TMA负载操作分两步实现。第一步是构造 TMA 复制描述符*主机代码*，而第二步是使用该描述符执行实际的 TMA 加载*内核代码。*请注意，这个两步过程与我们通常使用 CuTe 的 TiledCopy 所做的不同 - 所有复制步骤都写在内核代码中 - 如图所示[本教程](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/examples/cute/tutorial/tiled_copy.cu#L120-L124).

#### **主机代码**

在主机上，我们创建三个对象：我们复制的 GMEM 张量*从*，SMEM 张量的布局*每个*我们复制的 CTA 数量*进入*，和一个`tma_load`将这两个作为参数的对象。请注意，由于我们在主机上创建了 SMEM 布局，因此出于 TMA 负载的目的，所有 CTA 将共享相同的 SMEM 布局。一旦我们有了这些对象，它们就可以传递到设备上的内核，在内核中调用 TMA 加载操作。

主机上的整个代码块是：

```

template <typename T, int CTA_M, int CTA_N>
void host_fn(T* data, int M, int N) {
  using namespace cute;

  // create the GMEM tensor
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor = make_tensor(make_gmem_ptr(T), gmem_layout);

  // create the SMEM layout
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});

  // create the TMA object
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);

  // invoke the kernel
  tma_load_kernel<CTA_M, CTA_N>
                 <<<dim3{M / CTA_M, N / CTA_N, 1}, 1>>>
                 (tma_load, gmem_tensor, smem_layout);
}
```

这里的 `gmem_layout`、`gmem_tensor` 和 `smem_tensor` 仅用到了 CuTe 的基础概念，读者可先参考这些 [CuTe 教程 1](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/01_layout.md)、[教程 2](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/02_layout_algebra.md)、[教程 3](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/03_tensor.md)。这里的重点是 `tma_load` 对象：它是一个 `cute::TiledCopy` 实例，保存了执行 CTA 范围 copy 所需的信息与方法。示例中，`tma_load` 是通过 `cute::make_tma_copy` 的[显式默认配置](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/atom/copy_traits_sm90_tma.hpp#L1206-L1217)构造的。虽然该函数完整实现还有一些细节（本文后面讨论 `MULTICAST` 时会展开），但显式默认值已覆盖大多数场景，也更不容易出错。

让我们看看我们使用的签名`make_tma_copy`:

- 它的最后两个参数是`gmem_tensor`和`smem_layout`。在引擎盖下，`make_tma_copy`使用此信息来创建`TmaDescriptor`，这只是一个别名[CU张量图](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_desc.hpp#L178)。该描述符对象在 TMA 内核内部使用。
- 它的第一个参数是一个实例`SM90_TMA_LOAD`。该对象将复制操作分派到所需的位置`cp.async.bulk.tensor`PTX 调用，我们将在下面的第三部分更深入地讨论。

#### **内核代码**

相关的内核代码片段如下所示。这些行包含许多重要的 TMA 概念，我们将在下面解释。

```

template <typename T, int CTA_M, int CTA_N, class TmaLoad, class GmemTensor>
void tma_load_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor) {
  using namespace cute;
  constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);

  __shared__ T smem_data[CTA_M * CTA_N];
  __shared__ uint64_t tma_load_mbar;

  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);

  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));

    auto gmem_tensor_coord_cta = local_tile(
        gmem_tensor_coord,
        Tile<Int<CTA_M>, Int<CTA_N>>{},
        make_coord(blockIdx.x, blockIdx.y));

    initialize_barrier(tma_load_mbar, /* arrival count */ 1);

    set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);

    auto tma_load_per_cta = tma_load.get_slice(0);
    copy(tma_load.with(tma_load_mbar),
         tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
         tma_load_per_cta.partition_D(smem_tensor));
  }
  __syncthreads();
  wait_barrier(tma_load_mbar, /* phase */ 0);

  // after this line, the TMA load is finished
}
```

首先，在第 2 行，`tma_load`内核的参数必须用 __ 注释`grid_constant__ const`。如果我们有两个张量想要从 GMEM 复制到 SMEM，*每个*其中必须有自己的`TiledCopy`实例，并且每个实例必须是 __`grid_constant__ const`。这是传递的要求`cuTensorMap`如记录的那样从主机到设备[这里](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-tensor-memory-access-tma)， 例如。

下一个重要的一点是，对于 TMA 副本，只有一个线程负责发出 TMA 操作。在代码片段中，所有与TMA相关的变量和指令都包含在`if`从第 12 行开始的块，仅由线程 0 执行。另一方面，第 30 行包含 CTA 中所有线程等待 TMA 操作完成的指令。

##### 坐标和算术元组

现在，我们来看看 TMA 加载逻辑。这从第 13 行开始，我们在其中创建一个`gmem_tensor_coord`持有的物体*坐标*要复制的 GMEM 张量的。如果我们尝试以下操作：

```

if (cute::thread(0)) { cute::print(gmem_tensor_coord); }
```

然后我们看到像这样的输出（对于`M=N=1024`):

```

ArithTuple(_0,_0) o (1024,1024):(_1@1,_1@0)
```

对于熟悉 CuTe 中平铺复制工作方式的读者来说，第 15-18 行是不言自明的，其中 GMEM 张量被平铺为更小的分区，每个 CTA 根据块坐标切片到平铺张量中以获得其 GMEM 视图。但请注意，分区适用于上述`ArithTuple`代表坐标`gmem_tensor`，而不是`gmem_tensor`本身。特别是，`ArithTuple`被分割成形状的tile`[CTA_M,CTA_N]`，然后每个 CTA 获取其tile。

如果我们打印`gmem_tensor_coord_cta`使用`print_tensor`如下：

```

if (cute::block(7)) { cute::print_tensor(gmem_tensor_coord_cta); }
```

然后对于`CTA_M == CTA_N == 16`，我们看到：

```

ArithTuple(0,112) o (_16,_16):(_1@1,_1@0):
  (0,112)  (1,112)  (2,112)  (3,112)  (4,112)  (5,112)  (6,112)  (7,112)  (8,112)  (9,112)  (10,112)  (11,112)  (12,112)  (13,112)  (14,112)  (15,112)
  (0,113)  (1,113)  (2,113)  (3,113)  (4,113)  (5,113)  (6,113)  (7,113)  (8,113)  (9,113)  (10,113)  (11,113)  (12,113)  (13,113)  (14,113)  (15,113)
  // more lines
  (0,127)  (1,127)  (2,127)  (3,127)  (4,127)  (5,127)  (6,127)  (7,127)  (8,127)  (9,127)  (10,127)  (11,127)  (12,127)  (13,127)  (14,127)  (15,127)
```

这些数字是*坐标*在`gmem_tensor`其值将被复制到`smem_tensor`CTA 7. 我们鼓励读者在替换时尝试运行此代码片段`cute::block(7)`与其他索引一起了解哪些 CTA 从哪个坐标复制`gmem_tensor`.

接下来，第 25-27 行中发出的复制操作本身具有 TiledCopy 操作的常见签名，其中源张量被分区坐标替换。

##### 内存屏障

我们省略了第 20、22 和 30 行，所有这些都涉及`uint64_t`多变的`tma_load_mbar`它位于 SMEM 中。这是**异步事务屏障**我们用它来同步 TMA 加载与消耗加载到 SMEM 中的结果数据的内核的其余部分。 NVIDIA 中给出了此类屏障的高级描述[技术博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)在Hopper架构上。就我们的内核而言，重点如下：

1. 我们在第 20 行初始化共享内存中的 mbarrier 对象。 CuTe 方法`initialize_barrier`包装 PTX 指令`mbarrier.init.shared.b64`，它需要额外的*到达计数*范围。在我们的上下文中，由于单个线程将启动 TMA 负载，因此我们应该将到达计数设置为 1。此外，mbarrier 的起始阶段将始终设置为 0。
2. 我们都执行到达操作并使用 CuTe 方法在第 22 行设置 mbarrier 对象的预期事务计数`set_barrier_transaction_bytes`，它包装了 PTX 指令`mbarrier.arrive.expect_tx.shared::cta.b64`。事务计数设置为等于 TMA 负载传输的字节数，我们在第 4 行计算。
3. 第 25-27 行是复制指令，它将发送到所需的风味`cp.async.bulk.tensor`，总是有其完成机制`mbarrier::complete_tx::bytes`与提供的 mbarrier 对象。
4. 在第 30 行，我们对 mbarrier 对象执行等待操作。请注意，所有线程都在 mbarrier 上等待，而不是只有线程 0 到达 mbarrier，并且调用`__syncthreads()`之前有必要`wait_barrier`来解决线程分歧。  这里，`wait_barrier`包装 PTX 指令`mbarrier.try_wait.parity.shared::cta.b64`。这`try_wait`限定符（相对于`test_wait`) 表示等待是阻塞指令。这`parity`限定符的使用需要提供一个相位位，表示线程处于休眠状态，直到 mbarrier 的该相位位翻转。因为这是第一次使用 mbarrier 初始化后来跟踪完成情况，所以我们提供 0 作为阶段。如果我们要进行另一个 TMA 负载，则必须翻转相位才能重用 mbarrier。  一般来说，CUTLASS[管道API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/pipeline.md)在执行一系列 TMA 加载时，提供一种更高级别的方法来处理 mbarrier 对象的生命周期，就像在[软件流水线](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#pipelining)方案。
5. 后`wait_barrier`，内存一致性模型为我们提供了以下保证：由 TMA 加载完成的对 SMEM 的写入对于调用 mbarrier 等待的所有线程都是可见的（因此在我们的示例内核中，CTA 中的所有线程）。

##### REMAINDER TILES WITH TMA 和 STRIDE REQUIREMENTS

在上面的例子中，我们假设`m%CTA_M==0`和`n%CTA_N==0`。然而，为了执行 TMA 加载，我们可以完全放弃这个假设。当从 GMEM 加载剩余tile到 SMEM 时，TMA 复制单元必然需要自己处理越界逻辑[谓词](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0y_predication.md)内存复制不会越界读取。这与使用特殊的“隐式”CuTe 张量一致`ArithTuple`如上面 TMA 加载中所述 - 如果我们使用普通的 CuTe 张量，那么它们可以被切片以生成新的 CuTe 张量，其中可能存在指向 GMEM 的越界指针，这总是会导致错误。

然而，对于 TMA，GMEM 张量本身的步长有一个重要要求需要牢记，即**16 字节边界**要求。正如人们所预料的那样，TMA 不支持复制 GMEM 的任意跨步区域。相反，我们需要假设正在复制的tile具有 (i) 连续方向（步长 1），以及 (ii) 其他步长为 16 字节的倍数。这是[断言的](https://github.com/NVIDIA/cutlass/blob/7d49e6c7e2f8896c47f586706e67e1fb215529dc/include/cute/atom/copy_traits_sm90_tma.hpp#L846)在 CUTLASS 代码库中。

例如，对于我们的行主浮点数 GMEM 张量，其形状`(m, n)`并大步迈进`(n, 1)`，这提出了这样的要求：`n%4==0`。如果不满足这一点，那么可以在调用内核之前将输入张量填充到正确的范围。

## TMA 商店

掌握了 TMA 负载的基础知识后，由于两个操作之间有许多相似之处，学习 TMA 存储就容易多了。与TMA加载类似，实现TMA存储是一个两步过程：在主机上定义TMA复制描述符，然后在内核内部发出TMA存储操作。

#### 例如任务和代码

出于说明目的，让我们考虑 TMA 加载的反向示例，其中我们从多个 CTA 中的 SMEM 复制到分区 GMEM 张量中的相应tile。这里的区别在于，我们将在将 CTA 中的 SMEM tile复制到 GMEM 之前用简单的数字模式填充它们（否则，我们将复制未定义的值）。功能代码片段如下：

```

template <typename T, int CTA_M=32, int CTA_N=32>
void host_fn(T* data, int M, int N) {
  using namespace cute;

  // create the GMEM tensor
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor = make_tensor(make_gmem_ptr(T), gmem_layout);

  // create the SMEM layout
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});

  // create the TMA object
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);

  // invoke the kernel
  tma_store_kernel<CTA_M, CTA_N>
                  <<<dim3{M / CTA_M, N / CTA_N, 1}, CTA_M>>>
                  (tma_store, gmem_tensor, smem_layout);
}

template <typename T, int CTA_M, int CTA_N, class TmaStore, class GmemTensor>
void tma_store_kernel(__grid_constant__ const TmaStore tma_store, GmemTensor gmem_tensor) {
  using namespace cute;
  __shared__ T smem_data[CTA_M * CTA_N];

  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(T), smem_layout);

  // fill the rows of smem_data
  for (int j = 0; j < CTA_N; ++j) {
    smem_data(threadIdx.x, j) = threadIdx.x;
  }
 
  __syncthreads();
  tma_store_fence();

  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_store.get_tma_tensor(shape(gmem_tensor));

    auto gmem_tensor_coord_cta = local_tile(
      gmem_tensor_coord,
      Tile<Int<CTA_M>, Int<CTA_N>>{},
      make_coord(blockIdx.x, blockIdx.y));

    auto tma_store_per_cta = tma_store.get_slice(0);
    copy(tma_store,
         tma_store_per_cta.partition_S(smem_tensor),
         tma_store_per_cta.partition_D(gmem_tensor_coord_per_cta));
    // tma_store_arrive();
  }
  // tma_store_wait<0>();
}
```

主机代码看起来几乎与TMA负载的代码相同,除了调用`tma_store_kernel`。 Note that we have arranged for each CTA to have`CTA_M`线程。然后我们的示例让每个 CTA 持有一个`[CTA_M,CTA_N]`在 SMEM 中平铺，使得第 29-32 行中的线程`i`填充行`i`与价值`i`.

在内核代码中，`if`第 39-49 行中的块类似于`if`块在`tma_load_kernel`。特别是，只有线程`0`发出 TMA 存储操作。所有张量平铺逻辑在概念上都是相同的。但是，复制方向相反：对于 TMA 存储，`tma_store_per_cta.partition_S`方法应用于`smem_tensor`，而`tma_store_per_cta.partition_D`方法应用于 GMEM 张量的坐标。请注意，坐标也表示为`ArithTuple`，类似于TMA负载。

##### 记忆栅栏

TMA 加载和存储代码之间最重要的区别是我们不再看到任何 mbarrier 对象与 TMA 存储一起使用。这是因为 TMA 存储使用另一种机制来强制内存一致性：*记忆栅栏*.

内存栅栏的目的是在栅栏之前和之后执行线程请求的内存访问之间建立有保证的顺序。在我们的示例中，我们需要确保第 29-32 行中完成的所有对 SMEM 的写入对于线程 0 执行的 TMA 存储可见。为此，在第 35 行我们有 CuTe 方法`tma_store_fence()`包装 PTX 指令`fence.proxy.async.shared::cta`.

该指令包含两个重要的限定词来描述栅栏的效果：*范围*和*代理类*。这[范围](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scope)指示参与栅栏强制排序的线程集。在我们的例子中，限定符`cta`定义 CTA 中所有线程给定的范围（这是内存一致性模型的最小可能范围）。这[代理类](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies)指示除了通用代理之外，将参与栅栏强制执行的排序的代理类型。在我们的例子中，我们选择代理类型为`async.shared`因为 TMA 存储是在异步代理中执行的（相对于每个 CTA）。如果我们用不同的内存栅栏原语替换异步栅栏，例如`__threadfence_block()`如果不涉及异步代理，我们将破坏内核正确行为所需的保证，从而导致实践中的竞争条件。

##### TMA STORE ARRIVE AND WAIT

在第 49 行和第 51 行中，我们有`tma_store_arrive()`，它提交 TMA 存储操作（从技术上讲，作为`cp.async.bulk-group`）， 和`tma_store_wait<Count>()`，最多等待`Count`许多已提交的 TMA 存储操作正在挂起（例如，如果所有操作都应完成，则设置`Count`为 0)。当有其他内核工作等待 TMA 存储完成时，这些操作非常有用 - 例如，需要重用写出后可用的已释放 SMEM。但是，由于我们的内核在 TMA 存储完成后简单地退出，因此我们不需要 TMA 存储到达并等待模式，因此我们注释掉这些行。

## 深入了解 TMA 操作

TMA LOADTMA STORE方向GMEM -> SMEMSMEM -> GMEM同步方式内存屏障代理围栏何时同步手术后手术前TMA操作总结。

到目前为止，我们已经学习了如何调用 TMA 加载和 TMA 存储操作。上表对这些操作进行了比较和对比。要调用任一操作，我们需要创建一个类似于`TiledCopy`通过`cute::make_tma_copy`主机代码上的方法，然后将此对象传递给内核函数，我们在其中使用它们`cute::copy`实际调用该操作。在本节中，我们将更深入地探讨当我们调用这些时实际发生的情况`TiledCopy`核函数中的对象。通过这次深入研究，我们讨论了两个扩展：TMA 存储归约和 TMA 负载多播。

#### PTX TMA 加载和存储指令

PTX ([并行线程执行](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)）是 NVIDIA GPU 的低级中间语言。对于我们的讨论，PTX 的相关部分包含一组指令，这些指令可以通过由`asm volatile`关键词。特别是当我们调用`cute::copy(tma_load, ...)`或者`cute::copy(tma_store, ...)`如前几节所述，调用某些 PTX 指令来执行这些操作。通过研究PTX，我们可以更好地理解TMA加载和TMA存储。

让我们从 TMA 负载开始。回想一下，当我们创建`tma_load`在主机代码中的对象中，我们必须提供 GMEM 张量（其中包含要复制的源数据）和 SMEM 布局（描述数据在每个 CTA 内部的外观）。使用此张量和布局，CuTe 确定在以下情况下要执行的底层 PTX 指令：`cute::copy(tma_load, ...)`在内核中被调用。 PTX 指令的选择取决于*秩*GMEM 张量的（注意*秩*这里表示张量的维数，与线性代数中的矩阵 rank/nullity 相对）。在我们的示例中，GMEM 张量具有二阶，因此[以下 PTX 指令](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_tma.hpp#L100-L106)将被执行：

```

    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1)
      : "memory");
```

看看这条PTX指令，我们看到许多熟悉的概念。例如，`gmem_int_desc`指保存在 TMA 描述符中的坐标，而`mbarrier::complete_tx::bytes`和`smem_int_mbar`参考内存屏障。另请注意`tensor.2d`指的是我们正在复制一个 2 阶张量 即，一个 2D 矩阵。

事实证明，不仅 TMA 加载，所有 TMA 操作都是某些特定操作的包装器`cp.async.bulk`指示。这[NVIDIA PTX 文档专用于整个部分](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)讨论`cp.async.bulk`指令，特别是它们的语法和操作数。我们鼓励读者阅读该部分和其中的参考资料，以更深入地研究 TMA 操作，其涵盖的范围比本博客文章的预期范围大得多。在这里，我们将讨论通过这些公开的 TMA 的两个扩展`cp.async.bulk`指示。

#### TMA 商店减少

回想一下，TMA 将多个 CTA 的 SMEM 中的数据复制到 GMEM 张量中的相应tile中。我们可以*解释*TMA 存储为赋值操作，如以下 Python 伪代码所示：

```

for cta_idx in range(number_of_ctas):
  gmem_dst[cta_idx] = smem_src[cta_idx]
```

如果我们想要执行以下操作怎么办？

```

for cta_idx in range(number_of_ctas):
  gmem_dst[cta_idx] += smem_src[cta_idx]
  # or this:
  gmem_dst[cta_idx] = max(gmem_dst[cta_idx], smem_src[cta_idx])
  # or this:
  gmem_dst[cta_idx] = min(gmem_dst[cta_idx], smem_src[cta_idx])
```

所有这些操作——即减少总和、减少最大值和减少最小值——在张量程序中相当常见。特别是，reduce sum是Split-K GEMM中不可避免的子程序，而reduce max和reduce min则经常用于attention中。尽管这些操作看起来很简单，但在 CUDA 内核中实现它们并不是很简单。在阅读下一段之前，我们邀请读者简要思考一下 GMEM 和 SMEM 之间必须进行多少轮数据移动才能实现这些目标。

将 CTA 的 SMEM 中的值“累积”到 GMEM 张量中的tile中的归约操作的普通实现由一次 GMEM 读取、一个处理块和一次 GMEM 写入组成。首先，来自 GMEM 的原始值被加载到 CTA 的 SMEM 或寄存器中，然后进行归约操作，最后将结果写回。这个过程很慢。

对 TMA 存储的构造函数进行轻微修改`TiledCopy`对象允许我们将这个三步过程压缩为*只有一个*PTX指令，即`cp.reduce.async.bulk`而不是`cp.async.bulk`。准确地说，我们可以做如下*一行改变*在主机代码上：

```

// original: create a TMA store object
auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);

// to create a TMA reduce sum object
auto tma_reduce_sum = make_tma_copy(SM90_TMA_REDUCE_ADD{}, gmem_tensor, smem_layout);
```

然后使用`tma_reduce_sum`相反，现在调用`cp.reduce.async.bulk`而不是`cp.async.bulk`在引擎盖下。

顺便说一句，PTX 指令`cp.reduce.async.bulk`自CUDA 12.0发布以来就已经可用，但直到CUTLASS和CuTe才暴露出来[CUTLASS 3.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk)发布。我们希望在未来的版本中公开其他归约操作，但如果没有，则可以相当简单地将 CuTe 代码改编为 TMA reduce add 来执行最大和最小归约，以及其他按位归约`cp.reduce.async.bulk`优惠：`and`, `or`, `xor`, `inc`， 和`dec`.

#### TMA 加载组播

在上一节中，我们已经看到，研究 PTX 指令使我们能够发现 TMA 归约操作，对于某些应用程序，可以使用它来代替 TMA 存储。在本节中，我们将研究*组播*TMA负载的扩展。

为了帮助我们理解，我们首先看一下完整的语法[`cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor):

```

// global -> shared::cluster:
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism
{.multicast}{.level::cache_hint}
  [dstMem],
  [tensorMap, tensorCoords],
  [mbar]
  {, im2colOffsets}
  {, ctaMask}
  {, cache-policy}

.dst =                  { .shared::cluster }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.load_mode =            { .tile, .im2col }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster  }
```

同样，无需完全理解 PTX 指令的语法，我们就可以看到许多熟悉的概念，例如`.dim`, `.global`为了`src`， 和`.mbarrier`为了`completion_mechanism`。本节重点介绍`multicast`操作数。

多发射指一个情况,我们在GMEM索中有一个,我们想将其复制到.*多种的*SMEM 位于多个 CTA 中。这通常是 GEMM 内核（即，矩阵乘法）中的情况，其中多个行切片需要输入矩阵列切片，反之亦然。在这种情况下，虽然 TMA 负载仍然可以完美运行 - 我们只需向需要它的多个 CTA 提供相同的 TMA 描述符 -`.multicast`操作数允许我们*保证*L2 缓存命中。

让我们考虑将上述 TMA 负载示例扩展到多播负载示例。首先，我们需要定义*簇*我们内核的维度是不平凡的，因为 CTA 子集共同参与 TMA 负载多播操作的要求是它们属于同一个[（线程块）集群](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)。为了简单起见，我们只需更改网格尺寸，如下所示：

```

// old grid dimensions and implicit trivial cluster dimensions
dim3 grid_dims = dim3{M / CTA_M, N / CTA_N, 1};
dim3 cluster_dums = dim3{1, 1, 1};

// new grid dimensions and cluster dimensions
dim3 grid_dims = dim3{M / CTA_M, N / CTA_N, 2};
dim3 cluster_dums = dim3{1, 1, 2};
```

请注意，使用集群时，集群维度必须均匀划分为网格维度，否则内核将无法启动。在我们的新内核中，我们将为同一集群中的每对 CTA 安排相同的 GMEM tile加载到每个 CTA 的 SMEM 中，当且仅当两个 CTA 具有相同的值时才会发生这种情况。`blockIdx.x`和`blockIdx.y`.

首先，在主机代码中我们对TMA负载的定义进行以下更改`TiledCopy`目的：

```

// original: create a TMA load object
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);

// new: create a TMA load multicast object for the given cluster size
auto tma_load = make_tma_copy(SM90_TMA_LOAD_MULTICAST{},
      gmem_tensor, smem_layout, cute::_2{});
```

我们写`_2{}`对于最后一个参数（簇大小）将其作为编译时常量传递，使用[CuTe 整数类型](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md#integers)为此目的而提供。在实践中，更惯用的是，我们会定义`ClusterShape`事先输入（在我们的例子中，是`Shape<_1,_1,_2>`）然后写`size<2>ClusterShape{}`对于该参数。

然后我们将内核代码更改如下：

```

template <typename T, int CTA_M, int CTA_N, class ClusterShape,
          class TmaLoad, class GmemTensor>
void tma_load_kernel(__grid_constant__ const TmaLoad tma_load,
                     GmemTensor gmem_tensor) {
  using namespace cute;
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  constexpr uint32_t cluster_size = size<2>(ClusterShape{}));
  constexpr uint16_t tma_mcast_mask = (uint16_t(1) << cluster_size) - 1;
  constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);

  __shared__ T smem_data[CTA_M * CTA_N];
  __shared__ uint64_t tma_load_mbar;

  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(T), smem_layout);
  auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
  auto gmem_tensor_coord_cta = local_tile(
        gmem_tensor_coord,
        Tile<Int<CTA_M>, Int<CTA_N>>{},
        make_coord(blockIdx.x, blockIdx.y));

  if (threadIdx.x == 0) {
    initialize_barrier(tma_load_mbar, /* arrival count */ 1);
  }
  __syncthreads();
  cute::cluster_sync();
  cutlass::arch::fence_barrier_init();

  if (threadIdx.x == 0) {
    set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);
    auto tma_load_per_cta = tma_load.get_slice(block_rank_in_cluster);
    copy(tma_load.with(tma_load_mbar, tma_mcast_mask),
         tma_load_per_cta.partition_S(gmem_tensor_coord_per_cta),
         tma_load_per_cta.partition_D(smem_tensor));
  }
  __syncthreads();
  wait_barrier(tma_load_mbar, /* phase */ 0);

  // after this line, the TMA load is finished

  cute::cluster_sync();
}
```

我们已经强调了相关的变化。首先，我们现在需要跟踪 CTA 在其集群内的内部索引，我们通过 CuTe 方法获取该索引`block_rank_in_cluster()`。这将返回特殊寄存器的值`%cluster_ctarank`，在我们的示例中将采用值 0 和 1。为了简洁起见，我们将其称为`ctaid`。然后我们对要解压的代码进行以下三处修改：

1. 附加集群同步原语。
2. 使用`uint16`多播操作中的位掩码。
3. 使用`ctaid`在确定切片时`TiledCopy`用于划分 GMEM 和 SMEM 张量的对象。

对于(1)，我们使用CuTe方法`cluster_sync()`，这确实*两个都*集群屏障按顺序到达并等待操作。我们将其插入两个位置：在第 7-8 行中我们使用`cluster_sync()`与栅栏一起确保 mbarrier 初始化在集群范围内的可见性，在第 41 行我们使用另一个`cluster_sync()`确保集群中的两个 CTA 之一不会过早退出，而另一个仍在等待多播负载完成。一般来说，会对加载到SMEM中的数据进行计算，最后`cluster_sync()`将出现在内核代码的最末尾。

对于(2)，我们通过`uint16`位掩码到`copy`操作来指定哪些 CTA 将参与 TMA 多播负载。掩码中设置为 1 的位指示哪些 CTA 处于活动状态，集群中最多有 16 个 CTA（[最大非便携式尺寸](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#thread-block-clusters)）以及对应位的位置`ctaid`。因此，在我们的示例中，通过设置`tma_mcast_mask`到`0b11`我们指定集群中的两个 CTA 都将参与。

最后，对于(3)，`ctaid`用于指定从给定 CTA 启动的 TMA 多播加载操作切片到 GMEM 时使用的偏移量。为了清楚地解释这一点，请考虑以下加载 16 x 16 整数tile的示例，按升序行优先顺序初始化为 0-255，从 GMEM 到集群中两个 CTA 的 SMEM。假设我们错误地将 0 作为参数`tma_load.get_slice`为了**两个都**CTA。加载完成后，我们在两个 CTA 的 SMEM 中得到以下内容：

```

    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31
   32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47
   48   49   50   51   52   53   54   55   56   57   58   59   60   61   62   63
   64   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79
   80   81   82   83   84   85   86   87   88   89   90   91   92   93   94   95
   96   97   98   99  100  101  102  103  104  105  106  107  108  109  110  111
  112  113  114  115  116  117  118  119  120  121  122  123  124  125  126  127
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
```

相反，如果两个 CTA 的给定参数均为 1，那么我们会在两个 CTA 的 SMEM 中得到：

```

    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  128  129  130  131  132  133  134  135  136  137  138  139  140  141  142  143
  144  145  146  147  148  149  150  151  152  153  154  155  156  157  158  159
  160  161  162  163  164  165  166  167  168  169  170  171  172  173  174  175
  176  177  178  179  180  181  182  183  184  185  186  187  188  189  190  191
  192  193  194  195  196  197  198  199  200  201  202  203  204  205  206  207
  208  209  210  211  212  213  214  215  216  217  218  219  220  221  222  223
  224  225  226  227  228  229  230  231  232  233  234  235  236  237  238  239
  240  241  242  243  244  245  246  247  248  249  250  251  252  253  254  255
```

最后，给予*任何一个*0 来自`ctaid`1 和 1 来自`ctaid` 0, *或者*0 来自`ctaid`0 和 1 来自`ctaid`1，将整个tile正确加载到两个 CTA 的 SMEM。这些打印输出说明，从集群中的一个 CTA 发出多播操作，会将一半的 GMEM 加载到两个 CTA 的 SMEM 中的每一个中，其中`TiledCopy`确定各自的一半。这与组播的描述一致`cp.async.bulk.tensor`在 PTX 文档中：

> 源数据被多播到相同的 CTA 相对偏移量`dstMem`在每个目的地CTA的共享存储器中。

就`TiledCopy`对象，通常有一个布局`TiledLayout_TV`将线程值元组映射到tile的逻辑坐标，CuTe 将`ctaid`作为*线程索引*以达到切片的目的。例如，打印出`TiledCopy`在我们的 16 x 16 示例中，结果如下：

```

TiledCopy
  Tiler_MN:       (_16,_16)
  TiledLayout_TV: (_2,((_16,_16))):(_8,((_16,_1)))
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_256):(_0,_1)
  ValLayoutDst: (_1,_256):(_0,_1)
  ValLayoutRef: (_1,_256):(_0,_1)
  ValueType:    32b
```

它有两个“线程”对应集群中的两个CTA，偏移位置由逻辑坐标给出`(8,0)`在`(16,16)`tile用于`ctaid` 1.

## 结论

在这篇博文中，我们通过 TMA 加载、存储、存储归约和加载多播的几个简化示例，使用 CUTLASS 库提供的方法在 CUDA 内核中的 GMEM 和 SMEM 之间执行内存复制。

我们首先概述了 TMA，然后讨论了用户如何在 GPU 内核中调用这些操作。然后，我们深入研究低级 PTX 指令，以便更好地理解 TMA。我们希望这篇博文对想要了解 TMA、刷新有关该主题的知识或调试使用 TMA 的现有项目的读者有所帮助。

我们遗漏了一些重要的主题，例如 TMA 支持的混合模式以及 TMA 将 GMEM 复制到 SMEM 的能力**交错的**格式，在连续维度之外排列步幅。当将 TMA 与 Warpgroup 矩阵乘法累加 (WGMMA) 指令（也是 Hopper 架构的新指令）结合使用时，这些指令非常重要，以便以与 WGMMA 兼容的内存格式加载张量数据。当我们在以后的文章中讨论基于 Hopper 的 GEMM 时，我们将解释这些要点。

最后，可以在我们的网站上找到本博客文章中讨论的内核的完整示例[科尔法克斯研究 GitHub 存储库](https://github.com/ColfaxResearch/cfx-article-src/tree/master/tma).
