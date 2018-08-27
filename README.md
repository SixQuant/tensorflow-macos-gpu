> ![@老徐](http://oyiztpjzn.bkt.clouddn.com/avatar.png)老徐
>
> Thursday, 26 July 2018

#  Tensorflow 1.8 macOS GPU Install

> Tensorflow团队宣布停止支持1.2以后mac版的tensorflow gpu版本。
>
> 因此没办法直接安装只能自己用源码编译了。
>
> Tensorflow 1.8 with CUDA on macOS High Sierra 10.13.6

CPU 运行 Tensorflow 感觉不够快，想试试 GPU 加速！正好自己有一块支持CUDA的显卡。

![cpu-vs-gpu](https://github.com/SixQuant/tensorflow-macos-gpu/raw/master/res/cpu-vs-gpu.jpg)

## 版本

> 重要的事情说三遍：相关的驱动以及编译环境工具必须选择配套的版本，否则编译不成功！！！

版本：

- TensorFlow r1.8 source code，最新的1.9貌似还有问题
- macOS 10.13.6，这个应该关系不大
- 显卡驱动 387.10.10.10.40.105，支持的 CUDA 9.1
- CUDA 9.2，这个是 CUDA 驱动，可以高于上面的显卡支持的CUDA 版本，也就是 CUDA Driver 9.2
- cuDNN 7.2，与上面的CUDA对应，直接安装最新版
- **XCode 8.2.1**，这个是重点，请降级到这个版本，否则会编译出错或运行时出错 `Segmentation Fault`
- **bazel 0.14.0**，这个是重点，请降级到这个版本
- **Python 3.6**，这个是重点，不要使用最新版的 Python 3.7 截止目前编译会有问题

## 准备

需要下载（某些文件较大需要下载，请在继续阅读前先开始下载，节省时间）：

- Xcode 8.2.1

  https://developer.apple.com/download/more/

  Xcode_8.2.1.xip

- bazel-0.14.0

  https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-darwin-x86_64.sh


- CUDA Toolkit 9.2

  https://developer.nvidia.com/cuda-toolkit-archive

- cuDNN v7.2.1

  https://developer.nvidia.com/rdp/cudnn-download

- Tensorflow source code，333M

  ```Bash
  $ git clone https://github.com/tensorflow/tensorflow -b r1.8
  ```

### Python 3.6.5_1

目前装的是3.7，降级吧

```bash
$ brew unlink python
$ brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
$ pip3 install --upgrade pip setuptools wheel
# $ brew switch python 3.6.5_1
```

> 不要使用 Python 3.7.0，否则编译会有问题

编译完后可以切换回去

```bash
$ brew switch python 3.7.0
```

### Xcode 8.2.1

> 需要降级 Xcode 到 8.2.1

去apple开发者官网下载包，https://developer.apple.com/download/more/

解压后复制到`/Applications/Xcode.app`，然后进行指向

```bash
$ sudo xcode-select -s /Applications/Xcode.app
```

确认安装是否准确

```bash
$ cc -v
Apple LLVM version 8.0.0 (clang-800.0.42.1)
Target: x86_64-apple-darwin17.7.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

> Command Line Tools，cc 即 clang
>
> 这个很重要，否则虽然编译成功但是跑复杂一点项目会出现  `Segmentation Fault`

## 环境变量

> 由于用到 CUDA 的 lib 不是在系统目录下，因此需要设置环境变量来指向
>
> 在 Mac 下 LD_LIBRARY_PATH 无效，使用的是 DYLD_LIBRARY_PATH

配置环境变量编辑 `~/.bash_profile`或 `~/.zshrc`

```bash
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/extras/CUPTI/lib
export PATH=$CUDA_HOME/bin:$PATH
```

## 安装 CUDA

> CUDA是NVIDIA推出的用于自家GPU的**并行计算**框架，也就是说CUDA只能在NVIDIA的GPU上运行，**而且只有当要解决的计算问题是可以大量并行计算的时候才能发挥CUDA的作用。**

### 第一步：确认显卡是否支持 GPU 计算

> 在这里找到你的显卡型号，看是否支持
>
> https://developer.nvidia.com/cuda-gpus

我的显卡是 **NVIDIA GeForce GTX 750 Ti:**

| GPU                                                          | Compute Capability |
| ------------------------------------------------------------ | ------------------ |
| [GeForce GTX 750 Ti](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-750-ti) | 5.0                |

### 第二步：安装 CUDA

如果安装了其他版本的CUDA，需要卸载请执行

```Bash
$ sudo /usr/local/bin/uninstall_cuda_drv.pl
$ sudo /usr/local/cuda/bin/uninstall_cuda_9.1.pl
$ sudo rm -rf /Developer/NVIDIA/CUDA-9.1/
$ sudo rm -rf /Library/Frameworks/CUDA.framework
$ sudo rm -rf /usr/local/cuda/
```

> 为了万无一失，最好还是重启一下

首先需要说明的是：CUDA Driver 与 GPU Driver的版本必须一致，才能让CUDA找到显卡。

* GPU Driver 即显卡驱动
    * http://www.macvidcards.com/drivers.html

    * 我的 macOS 是 10.13.6 对应的驱动已经安装最新版 `387.10.10.10.40.105`

      https://www.nvidia.com/download/driverResults.aspx/136062/en-us

      ```
      Version:	387.10.10.10.40.105
      Release Date:	2018.7.10
      Operating System:	macOS High Sierra 10.13.6
      CUDA Toolkit:	9.1
      ```
* CUDA Driver 
    * http://www.nvidia.com/object/mac-driver-archive.html

    * 单独先安装 CUDA Driver，可以选择最新版本，看他对显卡驱动的支持

    * cudadriver_396.148_macos.dmg

      ```
      New Release 396.148
      CUDA driver update to support CUDA Toolkit 9.2, macOS 10.13.6 and NVIDIA display driver 387.10.10.10.40.105
      Recommended CUDA version(s): CUDA 9.2
      Supported macOS 10.13
      ```
* CUDA Toolkit
    * https://developer.nvidia.com/cuda-toolkit

    * 可以选择最新版本，这里选择 9.2

    * cuda_9.2.148_mac.dmg、cuda_9.2.148.1_mac.dmg


安装完成后检查：

```bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:08:12_CDT_2018
Cuda compilation tools, release 9.2, V9.2.148
```

确认驱动是否已加载

```Bash
$ kextstat | grep -i cuda.
  149    0 0xffffff7f838d3000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) E13478CB-B251-3C0A-86E9-A6B56F528FE8 <4 1>
```

测试CUDA能否正常运行：

```bash
$ cd /usr/local/cuda/samples
$ sudo make -C 1_Utilities/deviceQuery
$ ./bin/x86_64/darwin/release/deviceQuery
./bin/x86_64/darwin/release/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 750 Ti"
  CUDA Driver Version / Runtime Version          9.2 / 9.2
  CUDA Capability Major/Minor version number:    5.0
  Total amount of global memory:                 2048 MBytes (2147155968 bytes)
  ( 5) Multiprocessors, (128) CUDA Cores/MP:     640 CUDA Cores
  GPU Max Clock rate:                            1254 MHz (1.25 GHz)
  Memory Clock rate:                             2700 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 2097152 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.2, CUDA Runtime Version = 9.2, NumDevs = 1
Result = PASS
```

> 如果最后显示 Result = PASS，那么CUDA就工作正常

如果出现下列错误

```
The version ('9.1') of the host compiler ('Apple clang') is not supported
```

> 说明 Xcode 版本太新了，要求降级 Xcode
>

### 第三步：安装 cuDNN

> **cuDNN**（CUDA Deep Neural Network library）：是NVIDIA打造的针对深度神经网络的加速库，是一个用于深层神经网络的GPU加速库。如果你要用GPU训练模型，cuDNN不是必须的，但是一般会采用这个加速库。

cuDNN
- https://developer.nvidia.com/rdp/cudnn-download
- 下载最新版 cuDNN v7.2.1 for CUDA 9.2
- cudnn-9.2-osx-x64-v7.2.1.38.tgz

下好后直接把解压缩合并到CUDA目录/usr/local/cuda/下即可：

```Bash
$ tar -xzvf cudnn-9.2-osx-x64-v7.2.1.38.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib/libcudnn*
$ rm -rf cuda
```

### 第四步：安装 CUDA-Z

> 用来查看 CUDA 运行情况

```bash
$ brew cask install cuda-z
```
然后就可以从 Application 里运行 CUDA-Z 来查看CUDA运行情况了

![CUDA-Z](https://github.com/SixQuant/tensorflow-macos-gpu/raw/master/res/CUDA-Z.png)

## 编译

> 如果有已经编译好的版本，则可以跳过本章直接到"安装"部分

下面从源码编译 Tensorflow GPU 版本

### CUDA准备

> 请参考前面部分

### 编译环境准备

Python

```Bash
$ python3 --version
Python 3.6.5
```

> 不要使用 Python 3.7.0，否则编译会有问题

Python 依赖

```bash
$ pip3 install six numpy wheel
```

Coreutils，llvm，OpenMP

```Bash
$ brew install coreutils llvm cliutils/apple/libomp
```

Bazel

> 需要注意，这里必须是 0.14.0 版本，新或旧都能导致编译失败。下载0.14.0版本，[bazel发布页](https://github.com/bazelbuild/bazel/releases)
>

```Bash
$ curl -O https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-darwin-x86_64.sh
$ chmod +x bazel-0.14.0-installer-darwin-x86_64.sh
$ ./bazel-0.14.0-installer-darwin-x86_64.sh
$ bazel version
Build label: 0.14.0
```

> 太低版本可能会导致找不到环境变量，从而 Library not loaded

检查NVIDIA开发环境

```Bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:08:12_CDT_2018
Cuda compilation tools, release 9.2, V9.2.148
```

检查clang版本

```bash
$ cc -v
Apple LLVM version 8.0.0 (clang-800.0.42.1)
Target: x86_64-apple-darwin17.7.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

### 源码准备

拉取 TensorFlow 源码 release 1.8 分支并进行修改，使其与macOS兼容

这里可以直接下载修改好的源码

```Bash
$ curl -O https://raw.githubusercontent.com/SixQuant/tensorflow-macos-gpu/master/tensorflow-macos-gpu-r1.8-src.tar.gz
```

或者手工修改 

```Bash
$ git clone https://github.com/tensorflow/tensorflow -b r1.8
$ cd tensorflow
$ curl -O https://raw.githubusercontent.com/SixQuant/tensorflow-macos-gpu/master/patch/tensorflow-macos-gpu-r1.8.patch
$ git apply tensorflow-macos-gpu-r1.8.patch
$ curl -o third_party/nccl/nccl.h https://raw.githubusercontent.com/SixQuant/tensorflow-macos-gpu/master/patch/nccl.h
```

### Build

配置

```Bash
$ which python3
/usr/local/bin/python3
```

```Bash
$ ./configure
```

```
Please specify the location of python. [Default is /usr/local/opt/python@2/bin/python2.7]: /usr/local/bin/python3

Found possible Python library paths:
  /usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages]

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.2

Please specify the location where CUDA 9.1 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.2

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]3.0,3.5,5.0,5.2,6.0,6.1

Do you want to use clang as CUDA compiler? [y/N]:n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:

Do you wish to build TensorFlow with MPI support? [y/N]:
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
Configuration finished
```

> 一定要输入正确的版本
>
> * /usr/local/bin/python3
> * CUDA 9.2
> * cuDNN 7.2
> * compute capability 3.0,3.5,5.0,5.2,6.0,6.1 这个一定要去查你的显卡支持的版本，可以输入多个

上面实际上是生成了编译配置文件 `.tf_configure.bazelrc`

开始编译

```Bash
$ bazel clean --expunge
$ bazel build --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --action_env PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
```

> 编译过程中由于网络问题，可能会下载失败，多重试几次
>
> 如果bazel版本不对，可能会造成 DYLD_LIBRARY_PATH 没有传递过去，从而Library not loaded

#### 编译说明

--config=opt 的意思应该是

```
build:opt --copt=-march=native
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
```

> -march=native 表示使用当前CPU支持的优化指令来进行编译

查看当前 CPU 支持的指令集

```bash
$ sysctl machdep.cpu.features
machdep.cpu.features: FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH DS ACPI MMX FXSR SSE SSE2 SS HTT TM PBE SSE3 PCLMULQDQ DTES64 MON DSCPL VMX EST TM2 SSSE3 FMA CX16 TPR PDCM SSE4.1 SSE4.2 x2APIC MOVBE POPCNT AES PCID XSAVE OSXSAVE SEGLIM64 TSCTMR AVX1.0 RDRAND F16C
```

```bash
$ gcc -march=native -dM -E -x c++ /dev/null | egrep "AVX|SSE"

#define __AVX2__ 1
#define __AVX__ 1
#define __SSE2_MATH__ 1
#define __SSE2__ 1
#define __SSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#define __SSE_MATH__ 1
#define __SSE__ 1
#define __SSSE3__ 1
```

#### 编译错误 dyld: Library not loaded: @rpath/libcudart.9.2.dylib

```
ERROR: /Users/c/Downloads/tensorflow-macos-gpu-r1.8/src/tensorflow/python/BUILD:1590:1: Executing genrule //tensorflow/python:string_ops_pygenrule failed (Aborted): bash failed: error executing command /bin/bash bazel-out/host/genfiles/tensorflow/python/string_ops_pygenrule.genrule_script.sh
dyld: Library not loaded: @rpath/libcudart.9.2.dylib
  Referenced from: /private/var/tmp/_bazel_c/ea0f1e868907c49391ddb6d2fb9d5630/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/python/gen_string_ops_py_wrappers_cc
  Reason: image not found
```

> 是由于 bazel 的 bug 导致环境变量 DYLD_LIBRARY_PATH 没有传递过去

解决：安装正确版本的 bazel

#### 编译错误 PyString_AsStringAndSize

```
external/protobuf_archive/python/google/protobuf/pyext/descriptor_pool.cc:169:7: error: assigning to 'char *' from incompatible type 'const char *'
  if (PyString_AsStringAndSize(arg, &name, &name_size) < 0) {
```

> 这是因为 Python3.7 对 protobuf_python 有 bug, 请换为 Python3.6 后重新编译
>
> https://github.com/google/protobuf/issues/4086

编译时间长达1.5小时，请耐心等待

### 生成PIP安装包

重编译并且替换_nccl_ops.so

```Bash
$ gcc -march=native -c -fPIC tensorflow/contrib/nccl/kernels/nccl_ops.cc -o _nccl_ops.o
$ gcc _nccl_ops.o -shared -o _nccl_ops.so
$ mv _nccl_ops.so bazel-out/darwin-py3-opt/bin/tensorflow/contrib/nccl/python/ops
$ rm _nccl_ops.o
```

打包

```Bash
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/Downloads/
```

清理

```bash
$ bazel clean --expunge
```

## 安装

```bash
$ pip3 uninstall tensorflow
$ pip3 install ~/Downloads/tensorflow-1.8.0-cp36-cp36m-macosx_10_13_x86_64.whl
```

也可以直接通过http安装

```bash
$ pip3 install https://github.com/SixQuant/tensorflow-macos-gpu/releases/download/v1.8.0/tensorflow-1.8.0-cp36-cp36m-macosx_10_13_x86_64.whl
```

> 如果是直接安装，请一定要确认相关的版本是否和编译的一致或更高
>
> * cudadriver_396.148_macos.dmg
> * cuda_9.2.148_mac.dmg
> * cuda_9.2.148.1_mac.dmg
> * cudnn-9.2-osx-x64-v7.2.1.38.tgz

## 确认

> 确认 Tensorflow GPU 是否工作正常

### 确认环境变量

> 确认Python代码是否可以读取到正确的环境变量DYLD_LIBRARY_PATH

```bash
$ nano tensorflow-gpu-01-env.py
```

```python
#!/usr/bin/env python

import os

print(os.environ["DYLD_LIBRARY_PATH"])
```

```bash
$ python3 tensorflow-gpu-01-env.py
/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
```

### 确认是否启用了GPU

如果 TensorFlow 指令中兼有 CPU 和 GPU 实现，当该指令分配到设备时，GPU 设备有优先权。例如，如果 `matmul` 同时存在 CPU 和 GPU 核函数，在同时有 `cpu:0` 和 `gpu:0` 设备的系统中，`gpu:0` 会被选来运行 `matmul`。要找出您的指令和张量被分配到哪个设备，请创建会话并将 `log_device_placement` 配置选项设为 `True`。

```bash
$ nano tensorflow-gpu-02-hello.py
```

```python
#!/usr/bin/env python

import tensorflow as tf

config = tf.ConfigProto()
config.log_device_placement = True

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
with tf.Session(config=config) as sess:
    # Runs the op.
    print(sess.run(c))
```

```bash
$ python3 tensorflow-gpu-02-hello.py
2018-08-26 14:13:45.987276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: GeForce GTX 750 Ti major: 5 minor: 0 memoryClockRate(GHz): 1.2545
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 706.66MiB
2018-08-26 14:13:45.987303: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-08-26 14:13:46.245132: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 426 MB memory) -> physical GPU (device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0)
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0
2018-08-26 14:13:46.253938: I tensorflow/core/common_runtime/direct_session.cc:284] Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0

MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
2018-08-26 14:13:46.254406: I tensorflow/core/common_runtime/placer.cc:886] MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-08-26 14:13:46.254415: I tensorflow/core/common_runtime/placer.cc:886] b: (Const)/job:localhost/replica:0/task:0/device:GPU:0
a: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-08-26 14:13:46.254421: I tensorflow/core/common_runtime/placer.cc:886] a: (Const)/job:localhost/replica:0/task:0/device:GPU:0
[[22. 28.]
 [49. 64.]]
```

> 其中一些无用的看起来让人担心的日志输出我直接从源码中注释掉了，例如：
>
> OS X does not support NUMA - returning NUMA node zero
>
> Not found: TF GPU device with id 0 was not registered

### 跑复杂一点的

```bash
$ nano tensorflow-gpu-04-cnn-gpu.py
```

```python
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
import time
import numpy as np
import tflearn
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from tensorflow.python.client import device_lib
def print_gpu_info():
    for device in device_lib.list_local_devices():
        print(device.name, 'memory_limit', str(round(device.memory_limit/1024/1024))+'M', 
            device.physical_device_desc)
    print('=======================')

print_gpu_info()


DATA_PATH = "/Volumes/Cloud/DataSet"

mnist = tflearn.datasets.mnist.read_data_sets(DATA_PATH+"/mnist", one_hot=True)

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True

config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3

# Building convolutional network
net = tflearn.input_data(shape=[None, 28, 28, 1], name='input') 
net = tflearn.conv_2d(net, 32, 5, weights_init='variance_scaling', activation='relu', regularizer="L2") 
net = tflearn.conv_2d(net, 64, 5, weights_init='variance_scaling', activation='relu', regularizer="L2") 
net = tflearn.fully_connected(net, 10, activation='softmax') 
net = tflearn.regression(net,
                         optimizer='adam',                  
                         learning_rate=0.01,
                         loss='categorical_crossentropy', 
                         name='target')

# Training
model = tflearn.DNN(net, tensorboard_verbose=3)

start_time = time.time()
model.fit(mnist.train.images.reshape([-1, 28, 28, 1]),
          mnist.train.labels.astype(np.int32),
          validation_set=(
              mnist.test.images.reshape([-1, 28, 28, 1]),
              mnist.test.labels.astype(np.int32)
          ),
          n_epoch=1,
          batch_size=128,
          shuffle=True,
          show_metric=True,
          run_id='cnn_mnist_tflearn')

duration = time.time() - start_time
print('Training Duration %.3f sec' % (duration))
```

```bash
$ python3 tensorflow-gpu-04-cnn-gpu.py
2018-08-26 14:11:00.463212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: GeForce GTX 750 Ti major: 5 minor: 0 memoryClockRate(GHz): 1.2545
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 258.06MiB
2018-08-26 14:11:00.463235: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-08-26 14:11:00.717963: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/device:GPU:0 with 203 MB memory) -> physical GPU (device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0)
/device:CPU:0 memory_limit 256M
/device:GPU:0 memory_limit 204M device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0
=======================
Extracting /Volumes/Cloud/DataSet/mnist/train-images-idx3-ubyte.gz
Extracting /Volumes/Cloud/DataSet/mnist/train-labels-idx1-ubyte.gz
Extracting /Volumes/Cloud/DataSet/mnist/t10k-images-idx3-ubyte.gz
Extracting /Volumes/Cloud/DataSet/mnist/t10k-labels-idx1-ubyte.gz
2018-08-26 14:11:01.158727: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-08-26 14:11:01.158843: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 203 MB memory) -> physical GPU (device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0)
2018-08-26 14:11:01.487530: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-08-26 14:11:01.487630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 203 MB memory) -> physical GPU (device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0)
---------------------------------
Run id: cnn_mnist_tflearn
Log directory: /tmp/tflearn_logs/
---------------------------------
Training samples: 55000
Validation samples: 10000
--
Training Step: 430  | total loss: 0.16522 | time: 45.764s
| Adam | epoch: 001 | loss: 0.16522 - acc: 0.9660 | val_loss: 0.06837 - val_acc: 0.9780 -- iter: 55000/55000
--
Training Duration 45.898 sec
```

> 速度提升明显：
>
> CPU 版 无 AVX2 FMA，time: 168.151s
>
> CPU 版 加 AVX2 FMA，time: 147.697s
>
> GPU 版 加 AVX2 FMA，time: 45.898s

### cuda-smi

> cuda-smi 用来在Mac上代替 nvidia-smi

nvidia-smi是用来查看GPU内存使用情况的。

下载后放到 /usr/local/bin/ 目录下

```bash
$ sudo scp cuda-smi /usr/local/bin/
$ sudo chmod 755 /usr/local/bin/cuda-smi
$ cuda-smi
Device 0 [PCIe 0:1:0.0]: GeForce GTX 750 Ti (CC 5.0): 5.0234 of 2047.7 MB (i.e. 0.245%) Free
```

## 问题

### 错误 _ncclAllReduce

> 重新编译一个 _nccl_ops.so 复制过去即可

```bash
$ gcc -c -fPIC tensorflow/contrib/nccl/kernels/nccl_ops.cc -o _nccl_ops.o
$ gcc _nccl_ops.o -shared -o _nccl_ops.so
$ mv _nccl_ops.so /usr/local/lib/python3.6/site-packages/tensorflow/contrib/nccl/python/ops/
$ rm _nccl_ops.o
```

### Library not loaded: @rpath/libcublas.9.2.dylib

> 这是因为 Jupyter 中丢失了 DYLD_LIBRARY_PATH 环境变量
>
> 或者说是新版本的 MacOS 禁止了你对 DYLD_LIBRARY_PATH 等不安全因素的随意修改，除非你关闭SIP功能

重现

```python
import os
os.environ['DYLD_LIBRARY_PATH']
```

> 上面的代码在 Jupyter 中会出错，原因是因为 SIP的原因环境变量 DYLD_LIBRARY_PATH 不能被修改

解决：参考前面的 “环境变量” 设置部分

### Segmentation Fault

> 所谓的段错误就是指访问的内存超过了系统所给这个程序的内存空间

解决：请再次确认使用了正确的版本和编译参数，尤其是 XCode

### Not found: TF GPU device with id 0 was not registered

> 直接忽略这个警告

## GPU 内存有泄漏？？？

不知道咋解决:(

 