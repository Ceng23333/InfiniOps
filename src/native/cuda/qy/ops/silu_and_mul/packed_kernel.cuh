#ifndef INFINI_OPS_QY_SILU_AND_MUL_PACKED_KERNEL_CUH_
#define INFINI_OPS_QY_SILU_AND_MUL_PACKED_KERNEL_CUH_

#include "native/cuda/ops/silu_and_mul/kernel.cuh"

namespace infini::ops::qy {

template <typename T>
struct alignas(16) SiluPack {
  T elements[8];
};

// One block per row avoids per-element division and preserves scalar arithmetic.
template <typename T>
__global__ void SiluAndMulPackedKernel(T* out, const T* input, size_t hidden) {
  static_assert(sizeof(SiluPack<T>) == 16);
  const size_t row = blockIdx.x;
  auto gate = reinterpret_cast<const SiluPack<T>*>(input + row * 2 * hidden);
  auto up = reinterpret_cast<const SiluPack<T>*>(input + row * 2 * hidden + hidden);
  auto result = reinterpret_cast<SiluPack<T>*>(out + row * hidden);
  for (size_t column = threadIdx.x; column < hidden / 8; column += blockDim.x) {
    const auto g = gate[column];
    const auto u = up[column];
    SiluPack<T> value;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      value.elements[lane] = ::infini::ops::detail::SiluAndMulValue<Device::Type::kQy>(
          g.elements[lane], u.elements[lane]);
    }
    result[column] = value;
  }
}

}  // namespace infini::ops::qy

#endif
