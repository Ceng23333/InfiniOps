#ifndef INFINI_OPS_QY_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_QY_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"
#include <cstdlib>
#include <cstring>

namespace infini::ops {

template <>
struct ReshapeAndCacheFlashLaunchPolicy<Runtime<Device::Type::kQy>> {
  static int BlockSize() {
    const char *small_block = std::getenv("INFINI_QY_KV_SMALL_BLOCK");
    return small_block && std::strcmp(small_block, "1") == 0
               ? 128
               : RuntimeUtils<Device::Type::kQy>::GetOptimalBlockSize();
  }
};

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kQy>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kQy>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kQy>>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif
