#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

#define MAX_THREADS_NUM 512
#define MAX_BLOCKS_NUM 4096
#define BLOCK_NUM(count) min(((count + MAX_THREADS_NUM - 1) / MAX_THREADS_NUM), MAX_BLOCKS_NUM)
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

__global__ void matrix_array_set_kernel(int count,
                                        float *arr,
                                        float value) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    arr[index] = value;
  }
}

__global__ void matrix_broadcast_to_kernel(int inputCount, float* inputArr,
                                           int outputCount, float* outputArr) {
  CUDA_1D_KERNEL_LOOP(index, outputCount) {
      outputArr[index] = inputArr[index % inputCount];
  }
}

__global__ void matrix_reduce_sum_axis_zero_kernel(float* inputArr,
                                                   int outputCount, float* outputArr,
                                                   int zeroDim) {
      CUDA_1D_KERNEL_LOOP(index, outputCount) {
          float sum = 0;
          for (int i = 0; i < zeroDim; ++i) {
              sum += inputArr[index + i * outputCount];
          }
          outputArr[index] = sum;
      }
}

__global__ void matrix_elementwise_add_kernel(float* matAData, float* matBData,
                                              float* outputData, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputData[index] = matAData[index] + matBData[index];
    }
}

__global__ void matrix_elementwise_add_by_const_kernel(float* inputArr, float val,
                                                       float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] + val;
    }
}

__global__ void matrix_elementwise_multiply_kernel(float* matAData, float* matBData,
                                                   float* outputData, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputData[index] = matAData[index] * matBData[index];
    }
}

__global__ void matrix_elementwise_multipy_by_const_kernel(float* inputArr, float val,
                                                           float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] * val;
    }
}

__global__ void matrix_relu_kernel(float* inputArr, float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index];
        if (inputArr[index] < 0) {
            outputArr[index] = 0.f;
        }
    }
}

__global__ void matrix_relu_gradient_kernel(const float* inputArr, const float* gradArr,
                                            float* outputArr, int count) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        outputArr[index] = inputArr[index] > 0 ? gradArr[index] : 0;
    }
}

__global__ void matrix_softmax_kernel(int nRow, int nCol, float* inputArr, float* outputArr) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= nRow) return;

    float* input = inputArr + y * nCol;
    float* output = outputArr + y * nCol;

    float maxval = *input;
    for (int i = 1; i < nCol; ++i) {
        maxval = max(input[i], maxval);
    }
    float sum = 0;
    for (int i = 0; i < nCol; ++i) {
        sum += expf(input[i] - maxval);
    }
    for (int i = 0; i < nCol; ++i) {
        output[i] = expf(input[i] - maxval) / sum;
    }
}

/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) {
  int count = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    count *= arr->shape[i];
  }
  float *arr_data = (float *)arr->data;
  matrix_array_set_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
    count, arr_data, value);
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim + 1 == output->ndim);
  int inputCount = 1, outputCount = output->shape[0];
  for (int i = 0; i < input->ndim; ++i) {
      assert(input->shape[i] == output->shape[i + 1]);
      inputCount *= input->shape[i];
      outputCount *= output->shape[i + 1];
  }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_broadcast_to_kernel<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
    inputCount, inputArr, outputCount, outputArr);
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == output->ndim + 1);
  int zeroDim = input->shape[0], outputCount = 1;
    for (int i = 0; i < output->ndim; ++i) {
        assert(input->shape[i+1] == output->shape[i]);
        outputCount *= output->shape[i];
    }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_reduce_sum_axis_zero_kernel<<<BLOCK_NUM(outputCount), MAX_THREADS_NUM>>>(
          inputArr, outputCount, outputArr, zeroDim);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  assert(matA->ndim == output->ndim);
  assert(matB->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < matA->ndim; ++i) {
    assert(matA->shape[i] == output->shape[i]);
    assert(matB->shape[i] == output->shape[i]);
    count *= matA->shape[i];
  }
  float* matAData = (float*) matA->data;
  float* matBData = (float*) matB->data;
  float* outputData = (float*) output->data;
  matrix_elementwise_add_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          matAData, matBData, outputData, count);
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  assert(input->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
  }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_elementwise_add_by_const_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          inputArr, val, outputArr, count);
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  assert(matA->ndim == output->ndim);
  assert(matB->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < matA->ndim; ++i) {
    assert(matA->shape[i] == output->shape[i]);
    assert(matB->shape[i] == output->shape[i]);
    count *= matA->shape[i];
  }
  float* matAData = (float*) matA->data;
  float* matBData = (float*) matB->data;
  float* outputData = (float*) output->data;
  matrix_elementwise_multiply_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          matAData, matBData, outputData, count);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  assert(input->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    count *= input->shape[i];
  }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_elementwise_multipy_by_const_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          inputArr, val, outputArr, count);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  // Hint: use cublas
  // cublas assume matrix is column major
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(matC->ndim == 2);
  assert(matA->shape[transposeA ? 0 : 1] == matB->shape[transposeB ? 1 : 0]);
  assert(matA->shape[transposeA ? 1 : 0] == matC->shape[0]);
  assert(matB->shape[transposeB ? 0 : 1] == matC->shape[1]);

  cublasHandle_t handle;
  cublasCreate(&handle);
  const float* matAData = (const float*) matA->data;
  const float* matBData = (const float*) matB->data;
  float* matCData = (float*) matC->data;
  float alpha = 1, beta = 0;

  cublasSgemm(handle,
              (transposeB ? CUBLAS_OP_T : CUBLAS_OP_N),
              (transposeA ? CUBLAS_OP_T : CUBLAS_OP_N),
              (transposeB ? matB->shape[0] : matB->shape[1]),
              (transposeA ? matA->shape[1] : matA->shape[0]),
              (transposeB ? matB->shape[1] : matB->shape[0]),
              &alpha,
              matBData, matB->shape[1],
              matAData, matA->shape[1],
              &beta,
              matCData, (transposeB ? matB->shape[0] : matB->shape[1]));

  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < input->ndim; ++i) {
      assert(input->shape[i] == output->shape[i]);
      count *= input->shape[i];
  }
  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;
  matrix_relu_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          inputArr, outputArr, count);
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  assert(input->ndim == in_grad->ndim);
  assert(input->ndim == output->ndim);
  int count = 1;
  for (int i = 0; i < input->ndim; ++i) {
      assert(input->shape[i] == in_grad->shape[i]);
      assert(input->shape[i] == output->shape[i]);
      count *= input->shape[i];
  }
  const float* inputArr = (const float*) input->data;
  const float* gradArr = (const float*) in_grad->data;
  float* outputArr = (float*) output->data;
  matrix_relu_gradient_kernel<<<BLOCK_NUM(count), MAX_THREADS_NUM>>>(
          inputArr, gradArr, outputArr, count);
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0]);
  assert(input->shape[1] == output->shape[1]);

  int nRow = input->shape[0];
  int nCol = input->shape[1];

  dim3 block(MAX_THREADS_NUM);
  dim3 grid((nRow + block.x - 1) / block.x);

  float* inputArr = (float*) input->data;
  float* outputArr = (float*) output->data;

  matrix_softmax_kernel<<<grid, block>>>(nRow, nCol, inputArr, outputArr);

  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
