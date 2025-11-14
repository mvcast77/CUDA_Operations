# CUDA_Operations

### Collection of Operations like reductions and matrix multiplications, done with CUDA

*Under Construction*

## Basic Operations

gpucheck.cu

vector.cu

## Finished Operations, Listed by Complexity

matthrust.cu

reduction.cu

matcpy.cu

matmultTile.cu

matmultRealTile2.cu

## Instructions on Compiling and running

In order to compile any of these programs, the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) must be installed. Once this is done, check the compute capability of the GPU(s) in the system.

	make gpu

will compile the gpucheck.cu program, which provides information on the GPU(s) available. The Major/minor information printed is the compute capability of the device.

When using specific CUDA features, checking if the feature (and the version of CUDA it was released in) are supported by the compute capability of available devices will quickly diagnose compatability errors that may arise.
