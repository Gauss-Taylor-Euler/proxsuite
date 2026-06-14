#ifndef PROXSUITE_GQP_CONE_POSITIVE_ORTHANT
#define PROXSUITE_GQP_CONE_POSITIVE_ORTHANT

#include "proxsuite/proxgqp/dense/BaseGQP.hpp"

namespace proxsuite {
namespace proxgqp {
namespace dense {

template <typename T>
struct PositiveOrthantCone : Cone<T> {
  isize dimC;

  PositiveOrthantCone(isize dim) : dimC(dim) {}

  Mat<T> dualJacobian(VecRef<T> z) override {
    Mat<T> J = Mat<T>::Zero(dimC, dimC);
    for (isize i = 0; i < dimC; ++i) {
      if (z(i) > T(0)) {
        J(i, i) = T(1);
      }
    }
    return J;
  }

  Vec<T> dualProject(VecRef<T> z) override {
    return z.cwiseMax(T(0));
  }
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite

#endif
