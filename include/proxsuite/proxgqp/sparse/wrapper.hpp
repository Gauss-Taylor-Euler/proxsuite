#ifndef PROXSUITE_BASE_SPARSE_GQP
#define PROXSUITE_BASE_SPARSE_GQP
#include "BaseSparseGQP.hpp"
#include "proxsuite/proxgqp/dense/cones/LorentzCone.hpp"
#include "proxsuite/proxgqp/dense/fwd.hpp"
#include <iostream>
#include <ostream>

namespace proxsuite {
namespace proxgqp {
namespace sparse {
template <typename T> struct SparseGQP : BaseSparseGQP<T> {
  void test() {

    Vec<T> z(2);

    LorentzCone<T> cone(2);

    z.setRandom();

    std::cout << cone.dualSparseJacobian(z) << std::endl;

    z.setRandom();

    std::cout << z << std::endl;

    std::cout << "binding work" << std::endl;
  }
};
} // namespace sparse
} // namespace proxgqp

} // namespace proxsuite

#endif
