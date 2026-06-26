#ifndef PROXSUITE_BASE_SPARSE_GQP
#define PROXSUITE_BASE_SPARSE_GQP
#include "BaseSparseGQP.hpp"
#include <iostream>
#include <ostream>

namespace proxsuite {
namespace proxgqp {
namespace sparse {
template <typename T> struct SparseGQP : BaseSparseGQP<T> {
  void test() { std::cout << "binding work" << std::endl; }
};
} // namespace sparse
} // namespace proxgqp

} // namespace proxsuite

#endif
