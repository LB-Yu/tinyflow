/*!
 * \file device_api.h
 * \brief Device specific API
 */
#ifndef TINYFLOW_RUNTIME_CUDA_DEVICE_API_H_
#define TINYFLOW_RUNTIME_CUDA_DEVICE_API_H_

#include "c_runtime_api.h"
#include "device_api.h"
#include <cuda_runtime.h>

#include <assert.h>
#include <string>

namespace tinyflow {
namespace runtime {

class CUDADeviceAPI : public DeviceAPI {
public:
  void *AllocDataSpace(DLContext ctx, size_t size, size_t alignment) final;

  void FreeDataSpace(DLContext ctx, void *ptr) final;

  void CopyDataFromTo(const void *from, void *to, size_t size,
                      DLContext ctx_from, DLContext ctx_to,
                      DLStreamHandle stream) final;

  void StreamSync(DLContext ctx, DLStreamHandle stream) final;
};

} // namespace runtime
} // namespace tinyflow
#endif // TINYFLOW_RUNTIME_CUDA_DEVICE_API_H_
