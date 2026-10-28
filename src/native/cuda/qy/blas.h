#ifndef INFINI_OPS_QY_BLAS_H_
#define INFINI_OPS_QY_BLAS_H_

#include <utility>

// clang-format off
#include <cublas_v2.h>
// clang-format on

#include "data_type.h"
#include "native/cuda/blas.h"
#include "native/cuda/qy/blas_utils.h"
#include "native/cuda/qy/runtime_.h"

namespace infini::ops {

template <>
struct Blas<Device::Type::kQy> : public Runtime<Device::Type::kQy> {
  using BlasHandle = cublasHandle_t;

  static constexpr auto BLAS_OP_N = CUBLAS_OP_N;
  static constexpr auto BLAS_OP_T = CUBLAS_OP_T;
  static constexpr auto BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;

  static constexpr auto BlasCreate = cublasCreate;
  static constexpr auto BlasSetStream = cublasSetStream;
  static constexpr auto BlasDestroy = cublasDestroy;

  static constexpr auto BlasGemmStridedBatchedEx = [](
      cublasHandle_t handle, cublasOperation_t trans_a, cublasOperation_t trans_b,
      int m, int n, int k, const void *alpha, const void *a, cudaDataType a_type,
      int lda, long long stride_a, const void *b, cudaDataType b_type, int ldb,
      long long stride_b, const void *beta, void *c, cudaDataType c_type, int ldc,
      long long stride_c, int batch_count, cublasComputeType_t compute_type,
      cublasGemmAlgo_t algorithm) {
    // Preserve the installed QY provider's fast path for a single matrix.
    if (batch_count == 1) {
      return cublasGemmEx(handle, trans_a, trans_b, m, n, k, alpha, a, a_type,
                          lda, b, b_type, ldb, beta, c, c_type, ldc,
                          compute_type, algorithm);
    }
    return cublasGemmStridedBatchedEx(
        handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, stride_a,
        b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c,
        batch_count, compute_type, algorithm);
  };
};

}  // namespace infini::ops

#endif
