#ifndef INFINI_OPS_QY_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_QY_SILU_AND_MUL_KERNEL_H_

#include "native/cuda/ops/silu_and_mul/kernel.h"
#include "native/cuda/qy/caster.cuh"
#include "native/cuda/qy/runtime_.h"
#include "native/cuda/qy/ops/silu_and_mul/packed_kernel.cuh"

#include <cstdlib>
#include <cstring>

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kQy>
    : public CudaSiluAndMul<Runtime<Device::Type::kQy>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kQy>>::CudaSiluAndMul;

  void operator()(const Tensor input, Tensor out) const override {
    const char* packed = std::getenv("INFINI_QY_SILU_PACKED");
    const bool aligned = reinterpret_cast<uintptr_t>(input.data()) % 16 == 0 &&
                         reinterpret_cast<uintptr_t>(out.data()) % 16 == 0;
    if (packed && std::strcmp(packed, "1") == 0 && aligned &&
        is_input_contiguous_ && is_out_contiguous_ && hidden_size_ != 0 &&
        hidden_size_ % 8 == 0 && output_size_ / hidden_size_ >= 32 &&
        (out_type_ == DataType::kFloat16 || out_type_ == DataType::kBFloat16)) {
      DispatchFunc<ReducedFloatTypes>(
          {static_cast<int64_t>(out_type_)}, [&](auto tag) {
            using T = TypeMapType<Device::Type::kQy, ListGet<0>(tag)>;
            const auto threads = std::min<size_t>(hidden_size_ / 8, 768);
            auto stream = static_cast<Runtime<Device::Type::kQy>::Stream>(stream_ ? stream_ : 0);
            qy::SiluAndMulPackedKernel<T><<<output_size_ / hidden_size_, threads, 0, stream>>>(
                static_cast<T*>(out.data()), static_cast<const T*>(input.data()), hidden_size_);
          }, "QySiluAndMulPacked");
      return;
    }
    CudaSiluAndMul<Runtime<Device::Type::kQy>>::operator()(input, out);
  }
};

}  // namespace infini::ops

#endif
