#ifndef PROXSUITE_BASE_SPARSE_GQP_WITHOUT_SOLVE
#define PROXSUITE_BASE_SPARSE_GQP_WITHOUT_SOLVE
#include "fwd.hpp"
#include "proxsuite/proxgqp/dense/fwd.hpp"

namespace proxsuite {
namespace proxgqp {
namespace sparse {
template <typename T> struct BaseSparseGQP {
  BaseSparseGQP<T>(isize dim){};
};
} // namespace sparse
} // namespace proxgqp

} // namespace proxsuite

#endif
