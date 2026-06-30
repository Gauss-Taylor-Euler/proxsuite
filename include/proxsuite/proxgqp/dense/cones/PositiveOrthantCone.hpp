#ifndef PROXSUITE_GQP_CONE_POSITIVE_ORTHANT
#define PROXSUITE_GQP_CONE_POSITIVE_ORTHANT

#include "proxsuite/proxgqp/dense/BaseGQP.hpp"

namespace proxsuite {
namespace proxgqp {

template <typename T> struct PositiveOrthantCone : Cone<T> {
  isize dimC;

  PositiveOrthantCone(isize dim) : dimC(dim) {}

  SparseMat<T> dualSparseJacobian(VecRef<T> z) override {
    return convertToSparseMat(dualJacobian(z));
  }

  Mat<T> dualJacobian(VecRef<T> z) override {
    Mat<T> J = Mat<T>::Zero(dimC, dimC);
    for (isize i = 0; i < dimC; ++i) {
      if (z(i) > T(0)) {
        J(i, i) = T(1);
      }
    }
    return J;
  }

  Vec<T> dualProject(VecRef<T> z) override { return z.cwiseMax(T(0)); }

  Vec<T> applyJacobian(VecRef<T> arg, VecRef<T> x) override {
    return dualJacobian(arg) * x;
  }
  Mat<T> precondJacobian(VecRef<T>) override {
    return Mat<T>::Identity(dimC, dimC);
  }
  Mat<T> fastMultByDualJacobian(VecRef<T> z, MatRef<T> C) override {
    Mat<T> JC = Mat<T>::Zero(dimC, C.cols());
    for (isize i = 0; i < dimC; ++i) {
      if (z(i) > T(0)) {
        JC.row(i) = C.row(i);
      }
    }
    return JC;
  }
  Mat<T> fastMultByPrecondJacobian(VecRef<T>, MatRef<T> C) override {
    return C;
  }
};

} // namespace proxgqp
} // namespace proxsuite

#endif
