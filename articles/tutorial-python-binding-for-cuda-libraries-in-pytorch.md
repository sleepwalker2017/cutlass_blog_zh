---
title_zh: "教程：PyTorch 中 CUDA 库的 Python 绑定"
title_en: "Tutorial: Python bindings for CUDA libraries in PyTorch"
source_url: "https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/"
published_at: "2024-03-13"
source_site: "Colfax Research"
external: false
english_markdown: "articles-en/tutorial-python-binding-for-cuda-libraries-in-pytorch.en.md"
---
# 教程：PyTorch 中 CUDA 库的 Python 绑定

原文标题：Tutorial: Python bindings for CUDA libraries in PyTorch

英文对照：[articles-en/tutorial-python-binding-for-cuda-libraries-in-pytorch.en.md](../articles-en/tutorial-python-binding-for-cuda-libraries-in-pytorch.en.md)

PyTorch 是当今最流行的 AI 框架之一。它由 Meta（当时的 Facebook）开发，并于 2017 年开源，提供了易于上手、风格鲜明的 Pythonic 接口。这种易用性使它非常适合研究与开发场景，因为研究人员往往需要围绕新型 AI 工作负载不断迭代。不过，纯 Python 开发也有明显短板，其中最常见、也最关键的一项就是性能。如果 Python 代码完全不利用 GPU 硬件加速，或者只是以非常朴素的方式使用 GPU（例如没有针对特定 GPU 架构做优化），这一问题会尤为突出。

为了更充分地发挥 NVIDIA GPU 的性能，一个最直接的方法，就是让 PyTorch 调用那些已经过深度优化的 GPU 加速库。PyTorch 确实已经为许多常见 AI 工作负载集成了这类实现，但覆盖范围并不是无限的。对于某些特定工作负载，单独的 CUDA® C++ 库可能会比 PyTorch 默认使用的实现更快。

反过来说，如果某位 CUDA 开发者写出了一个新的高性能库，他也往往会希望把它接入 PyTorch，从而降低使用门槛。虽然像 PyCUDA 这样的工具已经能让 Python 调用 CUDA，但 CUDA 开发的主战场依然是 C++。因此，开发者通常需要把 C++ 实现包装成一个可供 PyTorch 使用的 Python 接口。

PyTorch 官网已经有一篇非常有帮助的[指南](https://pytorch.org/tutorials/advanced/cpp_extension.html)，详细介绍了如何编写 C++ 扩展。本文会在此基础上补充一些实践中非常有用的信息，尤其适用于 CUDA 以及诸如 [CUTLASS](https://github.com/NVIDIA/cutlass/) 这类 CUDA 库。作为示例，我们将实现一个 PyTorch C++ 扩展，利用 NVIDIA 的 CUTLASS 库完成通用矩阵乘法（GEMM），并把 Python 侧接口设计得与 [torch.mm](https://pytorch.org/docs/stable/generated/torch.mm.html) 类似，以便它能作为直接替代品使用。我们的目标是给出一个完整、可运行、便于后续扩展的模板。

## 将输入从 Torch 转换为 CUTLASS

我们的实现将基于 CUTLASS 的 `basic_gemm` [示例 0](https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu)。熟悉 CUTLASS 的读者会注意到，这个示例使用的是 2.x 语法。本文附录中还会给出一个面向 NVIDIA Hopper™ 架构、基于 3.x 语法的单独版本。

首先，为了简化，我们以该示例为例并将其包装在单个函数调用中：

```

template<typename DataType, typename OutputType>
void cutlass_gemm_wrapper(int M, int N, int K,
                          DataType const* ptrA, 
                          DataType const* ptrB,
                          OutputType* ptrC);
```

然后我们来看这次调用所需的参数。具体来说，需要准备三样东西：

1. 张量的形状，
2. 张量的数据类型，以及
3. 指向数据的指针。

我们的目标是编写一个函数：它从 PyTorch 接收输入，提取上述信息，再调用 CUTLASS 的包装函数。

### 输入 Torch 张量

新函数 `cutlass_gemm` 的输入参数将采用 `torch::Tensor` 类型，它是 Python 中 `torch.Tensor` 在 C++ 侧的表示。例如，该函数可以写成：

```

torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C)
```

注意，在上面的代码里，矩阵 `C` 仍然被保留为必选参数，尽管在 `torch.mm` 中它其实是可选的。这个问题我们会在后文处理。

### 张量形状

为了提取 GEMM 所需的信息，我们可以使用 PyTorch 的 [ATen API](https://pytorch.org/cppdocs/notes/tensor_basics.html)。首先，可以通过 `.sizes()` 方法获取张量形状：

```

auto A_shape = A.sizes();
```

它会返回一个数组（更准确地说，是 Torch 的 `IntArrayRef`），其中保存了张量的形状信息。

### 张量数据类型

接下来是数据类型。Torch 张量可能具有多种数据类型，可以通过 `.dtype()` 方法获取：

```

auto A_type = A.dtype();
```

然后可以将其与 Torch 数据类型进行比较：

```

bool is_half = (A.dtype() == torch::kHalf);
```

可以找到不同数据类型的完整列表[这里](https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h).

### 张量数据指针

最后，我们可以通过张量的 `.data_ptr()` 方法取得底层数据指针：

```

float* A_ptr = A.data_ptr<float>();
```

这里的 `.data_ptr()` 是模板化接口，允许开发者将返回的指针解释为所需的数据类型。请注意，如果您的应用程序仅处理默认数据类型，但不支持自定义数据类型，则此模板就足够了。例如，在 CUTLASS 中，FP16 数据类型为`cutlass::half_t`，而对应的 FP16 数据类型`.data_ptr()`被模板化的是`torch::kFloat16`.

因此，我们不使用模板，而是使用`reinterpret_cast`转换为我们需要的数据类型：

```

float* A_ptr = reinterpret_cast<float*>(a.data_ptr());
```

对于我们的示例，在这个示例中，我们让 CUTLASS 使用用户输入的数据类型。因此，我们可以使用在上一步中找到的数据类型来转换为正确的精度。为此，我们将`reinterpret_cast`在中间函数内并使用 C++ 模板来传递数据类型。

```

template<typename DataType, typename OutputType>
void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Get data sizes
  const int M = A.sizes()[0];
  const int K = B.sizes()[0];
  const int N = B.sizes()[1];

  // Casting to the data type of the input tensor
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  DataType *ptrC = reinterpret_cast<OutputType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType, OutputType>(M, N, K, ptrA, ptrB, ptrC);
}
```

请注意，模板参数在编译时解析，但这里我们需要选择正确的模板实例化`cutlass_gemm_unpack`基于数据类型`A`和`C`，我们在运行时知道。为此，我们可以引入一些条件逻辑，例如：

```

if(A.dtype() == torch::kFloat16 && C.dtype() == torch::kFloat32)
    cutlass_gemm_unpack<cutlass::half_t,float>(A, B, C);
// ...
```

事实上，我们并不完全是这样写代码的。在讨论了一些更重要的点之后，我们将进一步展示完整的程序。

## 输入验证

现在我们已经拿到了输入及其元数据，下一步是做输入合法性检查。基于张量形状和 dtype 的基础检查（例如矩阵乘法维度兼容性）比较直接，因此这里重点讨论 Torch 与 CUTLASS 相关的几个关键约束。

CUTLASS 对矩阵乘法的一个限制是它必须是连续的，这意味着相邻元素在内存中也是相邻的。由于 PyTorch 张量是行优先的，因此连续张量是指同一行和相邻列中的元素在内存中彼此相邻的张量。我们可以检查张量是否与`.is_contiguous()`方法。

```

bool a_contiguous = A.is_contiguous();
```

如果张量不连续，可以使用以下方法使它们连续`.contiguous()`方法。

```

torch::Tensor _A = A.contiguous();
```

如果原始张量已经是连续的，则此方法仅返回原始张量。但是，如果不是，它会创建一个新的连续张量。对于输入矩阵来说这不是问题`A`和`B`，但是对于`C`矩阵这是一个问题，因为`torch.mm`支持就地操作。因此，对于 C 矩阵，如有必要，我们将使用以下命令将数据复制回来`.copy_()`.

```

torch::Tensor _C = C.contiguous();

// ... GEMM operation ... //
 
if(!C.is_contiguous())
    C.copy_(_C);
return C
```

另一个限制是数据必须位于 GPU 器件上。我们可以通过以下方式轻松检查：

```

bool is_cuda = A.device().is_cuda();
```

我们的库专为 GPU 构建。如果数据必须在主机上分配，我们使用以下命令将其移动到 Python 中的设备`.to()`方法。虽然可以使用以下命令自动将数据移动到设备`.to()`在 C++ 中，此行为与大多数其他 PyTorch 函数不一致，因此如果设备不是 GPU，我们将抛出错误。

## 使 C 成为可选

喜欢PyTorch的`mm`，我们的函数将返回`C`张量返回到 PyTorch 以便在那里使用。我们还需要更新函数参数来标记`C`作为可选的。 Torch C++ API 提供了一个实用程序`c10::optional<torch::Tensor>`以便将 Tensor 参数指定为可选。有了这个，我们可以检查输入是否是通过`.has_value()`方法。如果这返回`true`，然后我们可以得到这个值`.value()`方法。

如果`.has_value()`返回`false`，那么我们需要创建一个新的张量。 ATen 有很多创建张量的选项，这些选项都有记录[这里](https://pytorch.org/cppdocs/notes/tensor_creation.html)。出于我们的目的，我们只需要一个空张量。结合起来，我们得到：

```

torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> out) { 

  // Handling the optional C matrix
  torch::Tensor C;
  if(out.has_value()) {  // Output tensor was provided. So we will use it.
    C = out.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];

    // We will allocate the matrix on GPU and set the datatype to be the same as the input
    auto c_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
    C = torch::empty({M, N}, c_options);
  }

  // ... Rest of the GEMM workload ...//
}
```

创建新矩阵时，我们设置选项将设备设置为GPU，数据类型与输入Tensor相同。创建新张量时建议使用 ATen 库。虽然可以创建一个新的`torch::Tensor`从现有的数据指针，这意味着 ATen 不拥有该数据。这可能会限制某些操作，例如张量传回 Python 后调整大小。因此，虽然 CUTLASS 有特殊的分配器，例如`HostTensor`，我们不会使用它们。

## 把它放在一起

最后，将所有内容放在一起，我们得到：

```

template<typename DataType, typename OutputType>
void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Get data sizes
  const int M = A.sizes()[0];
  const int K = B.sizes()[0];
  const int N = B.sizes()[1];

  // Casting to the data type of the input tensor
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  DataType *ptrC = reinterpret_cast<OutputType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType, OutputType>(M, N, K, ptrA, ptrB, ptrC);
}

// Intermediate function to get the output precision to use for the wrapper template. 
template<typename DataType>
void cutlass_gemm_find_output_type(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  if(C.dtype() == torch::kFloat16)
    cutlass_gemm_unpack<DataType, cutlass::half_t>(A, B, C);
  else if(C.dtype() == torch::kFloat32)
    cutlass_gemm_unpack<DataType, float>(A, B, C);
  else
    throw std::invalid_argument("Unsupported precision type");
} 

// This function is bound to "cutlass_gemm.mm". Takes torch::Tensors as inputs
torch::Tensor cutlass_gemm(torch::Tensor A,  // A matrix (m x k)
                           torch::Tensor B,  // B matrix (k x n)
                           c10::optional<torch::Tensor> out) {  // optional out matrix (m x n)
  // Handling the optional C matrix
  torch::Tensor C;
  if(out.has_value()) {  // Output tensor was provided. So we will use it.
    C = out.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];
    // We will allocate the matrix on GPU and set the datatype to be the same as the input
    auto c_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
    C = torch::empty({M, N}, c_options);
  }

  // Check that all tensors are allocated on GPU device.
  if(!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device.
                                 Use .to(device=torch.device('cuda'))");

  // Ensuring that the matrices are contiguous. 
  torch::Tensor _A = A.contiguous();
  torch::Tensor _B = B.contiguous();
  torch::Tensor _C = C.contiguous();

  // Select the CUTLASS precision type to use based on Torch input data type.
  if(A.dtype() == torch::kFloat16)
    cutlass_gemm_find_output_type<cutlass::half_t>(_A, _B, _C);
  else if(A.dtype() == torch::kFloat32)
    cutlass_gemm_find_output_type<float>(_A, _B, _C);
  else
    throw std::invalid_argument("Unsupported precision type");

  // If C was not contiguous, C != _C so copy the result back into C
  if(!C.is_contiguous())
    C.copy_(_C);

  // Return the Torch tensor back to PyTorch
  return C;
}
```

在此代码中，我们采用了一种临时方法来处理基于数据类型分派到适当模板化函数所需的条件逻辑。`A`和`C`。显然，这不能很好地扩展到大量模板参数。有关如何使用 Python 脚本来处理为高度模板化的 CuTe/CUTLASS 函数（如 CUTLASS 中的函数）编写包装器的示例，我们建议查看[_python_gemm](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass/emit/pytorch.py#L704)方法和[EmitGemmUniversalInstance3x](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass/backend/gemm_operation.py#L1195)CUTLASS 库中的类。

## 绑定与编译

函数实现完成后，下一步是编译并绑定到 Python。我们使用 `PyBind11` + `setuptools` 完成这一过程。这里不会展开工具链细节，只讲与本文实现直接相关的部分。

### pybind11

我们函数的绑定是：

```

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm", 
        py::overload_cast<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>(
          &cutlass_gemm), 
        py::arg("A"), 
        py::arg("B"), 
        py::arg("out") = py::none());
}
```

我们还将第三个参数指定为关键字参数“out”，与`torch.mm`，并将其设置为默认为 Python`None`.

### 设置工具

不幸的是，开箱即用`setuptools`不支持`nvcc`，CUDA 编译器。虽然有解决方法，但它可能相当[复杂的](https://stackoverflow.com/questions/10034325/can-python-distutils-compile-cuda-code)。 幸运的是，PyTorch 附带了一个名为`CUDAExtension`可以编译CUDA代码。

```

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

### ... set up lists cutlass_include_dirs, nvcc_flags, and ld_flags ... ###
setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(name="cutlass_gemm",
                      sources=["cutlass_gemm.cu"],
                      include_dirs=cutlass_include_dirs,
                      extra_compile_args={'nvcc': nvcc_flags},
                      libraries=ld_flags)
    ],
    cmdclass={'build_ext': BuildExtension})
```

参数的语法与扩展基类相同。但是，它会自动为 Torch 库添加所有必要的标志。 所以我们唯一要做的就是添加 CUTLASS 的路径。并且因为 CUTLASS 是一个仅包含头文件的库，所以我们只需要设置`include_dir`。一旦你运行`setup.py`，我们现在有了新模块`cutlass_gemm`可通过我们的 PyTorch 代码访问。

## 呼唤我们的新`mm`与 PyTorch

这是一个简单的 PyTorch 脚本，它使用我们的新函数执行 CUTLASS GEMM 操作。

```

import math
import cutlass_gemm

M = K = N = 4096
cuda = torch.device('cuda')
A = torch.normal(0,1,size=(M, K)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(K)
B = torch.normal(0,1,size=(K, N)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(K)

C1 = cutlass_gemm.mm(A,B)
print("cutlass_gemm.mm result:")
print(C1)
print()

C2 = torch.mm(A,B)
print("torch.mm result:")
print(C2)
print()
print("max deviation: {:.10f}".format(torch.max(torch.abs(C2-C1))))
```

我们指定`.to(device=cuda)`使`A`和`B`可通过 GPU 访问，并且我们对两个矩阵使用 FP16 精度。此外，我们还有一个验证步骤`torch.mm`显示与 Torch 版本的最大偏差。

```

cutlass_gemm.mm result:
tensor([[-0.0045, -0.0139,  0.0109,  ...,  0.0192, -0.0117,  0.0083],
        ...,
        [ 0.0110,  0.0005, -0.0079,  ...,  0.0106, -0.0012, -0.0083]],
       device='cuda:0', dtype=torch.float16)

torch.mm result:
tensor([[-0.0045, -0.0139,  0.0109,  ...,  0.0192, -0.0117,  0.0083],
        ...,
        [ 0.0110,  0.0005, -0.0079,  ...,  0.0106, -0.0012, -0.0083]],
       device='cuda:0', dtype=torch.float16)

max deviation: 0.0000610352
```

在这里,我们可以看到,结果矩阵实际上使用了FP16精度格式,我们得到相同的结果 (在epsilon内)`torch.mm`。 So now we can use this optimized GEMM in place of`torch.mm`.

## 代码下载

完整示例的源代码可以在[科尔法克斯研究 github](https://github.com/ColfaxResearch/cfx-article-src).

## 附录 A：AMP 支持

PyTorch有一个功能叫做[自动混合精度（AMP）](https://pytorch.org/docs/stable/amp.html)可用于简化混合精度工作负载。它围绕`autocast`上下文，其中操作在适当时自动使用较低的精度。这可以显着提高性能。

我们的示例不支持此功能，但是您可以在 C++ 包中找到有关 AMP 支持的更多信息[这里](https://pytorch.org/tutorials/advanced/dispatcher.html#autocast).

## 附录 B：CUTLASS 3.X 和 Hopper 架构

如前所述，上面的示例对 CUTLASS 使用 2.X 语法。在我们的存储库中，我们还提供了一个基于以下内容的 CUTLASS 3.X 示例`hopper_warp_specialized_gemm` [实施例48](https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu)。但是，在本文的范围内，2.X 和 3.X CUTLASS 所需的内容没有区别。我们的 3.X 示例仍然将所有 CUTLASS 代码包装在包装函数中。有关 CUTLASS 3.X 以及如何针对特定架构进行优化的更多信息，请参阅 CUTLASS 文档。

## 附录 C：构建后端

在本文中，我们的重点是编写可与 PyTorch 一起使用的扩展。为此，我们使用了`setuptools`作为与 PyTorch 结合的构建后端`CUDAExtension`实用类。然而，这将 PyTorch 添加为我们的扩展的依赖项，如果扩展不是为 PyTorch 开发的，这可能并不理想。可以使用`setuptools`无需依赖`CUDAExtension`。 有关示例，请参阅 python 安装[CUTLASS](https://github.com/NVIDIA/cutlass/tree/main).

此外，还有其他兼容的构建后端`nvcc`可用于创建基于 C/CuTe 的 Python 扩展。例如，[scikit 构建核心](https://github.com/scikit-build/scikit-build-core)是一个基于 cmake 的后端，可以用来代替`setuptools`。 有使用指南`nvcc`在`cmake`于[Nvidia 开发者论坛](https://developer.nvidia.com/blog/building-cuda-applications-cmake/).

最后一点，构建后端通常在`pyproject.toml`然后由 python 打包软件使用的文件。详细信息`pyproject.toml`和它的用法可以找到[这里](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
