Tinyflow is a simple deep learning framework for learning purposes. It supports automatic 
differentiation and GPU acceleration. TinyFlow currently provides all the operators needed 
to build a multilayer perceptron models (MLP).

If you want to learn more about the principles behind Tinyflow, the following two blog posts may provide a lot of intuition.
+ [Automatic Differentiation Based on Computation Graph](https://lb-yu.github.io/2019/07/22/Automatic-Differentiation-Based-on-Computation-Graph/)
+ [Tinyflow - A Simple Neural Network Framework](https://lb-yu.github.io/2019/07/23/Tinyflow-A-Simple-Neural-Network-Framework/)

# Install
Tinyflow currently only supports running in 64-bit linux environment. Requirement:
+ gcc >= 4.8;
+ cmake >= 3.13 (if you choose to use cmake);
+ CUDA 9.0
+ python 3

Download the source code.
```shell
git clone https://github.com/LB-Yu/tinyflow.git
```

Generally speaking, CUDA will be installed in `/use/local/cuda`. 
If your installation path is different, please modify the `CUDA_DIR` variable on the first 
line of the Makefile to your installation path, or modify the `CUDA_DIR` variable on the 
fourth line of CMakeLists.txt to your installation path.

For compiling with Makefile.
```shell
cd tinyflow
make
```

For compiling with CMake.
```shell
cd tinyflow
mkdir build
cmake ..
make
make install
```

# Run the MNIST Example
After compiling the GPU library, we can train an MLP on the MNIST dataset.
```shell
export PYTHONPATH="/path/to/tinyflow/python:${PYTHONPATH}"

# see cmd options with 
# python tests/mnist_dlsys.py -h

# run logistic regression on numpy
python tests/mnist_dlsys.py -l -m logreg -c numpy
# run logistic regression on gpu
python tests/mnist_dlsys.py -l -m logreg -c gpu
# run MLP on numpy
python tests/mnist_dlsys.py -l -m mlp -c numpy
# run MLP on gpu
python tests/mnist_dlsys.py -l -m mlp -c gpu
```

# Overview of Module
- python/dlsys/autodiff.py: Implements computation graph, autodiff, GPU/Numpy Executor.
- python/dlsys/gpu_op.py: Exposes Python function to call GPU kernels via ctypes.
- python/dlsys/ndarray.py: Exposes Python GPU array API.

- src/dlarray.h: header for GPU array.
- src/c_runtime_api.h: C API header for GPU array and GPU kernels.
- src/gpu_op.cu: cuda implementation of kernels
