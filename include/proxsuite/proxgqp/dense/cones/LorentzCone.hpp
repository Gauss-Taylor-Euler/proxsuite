#ifndef PROXSUITE_GQP_CONE_LORENTZ
#define PROXSUITE_GQP_CONE_LORENTZ

#include "proxsuite/proxgqp/dense/BaseGQP.hpp"
#include <cmath>

namespace proxsuite {
namespace proxgqp {
namespace dense {

template <typename T> struct LorentzCone : Cone<T> {
  isize dimC;

  LorentzCone(isize dim) : dimC(dim) {}

  Vec<T> dualProject(VecRef<T> z) override {

    isize n = dimC - 1;
    auto a = z.head(n);
    T b = z(n);

    T aNorm = a.norm();

    Vec<T> out(dimC);
    if (aNorm <= b) {
      out = z;
    } else if (aNorm <= -b) {
      out.setZero();
    } else {
      T coeff = (b + aNorm) / (T(2) * aNorm);
      out.head(n) = coeff * a;
      out(n) = coeff * aNorm;
    }
    return out;
  }

  Mat<T> dualJacobian(VecRef<T> z) override {
    isize n = dimC - 1;
    auto a = z.head(n);
    T b = z(n);

    T aNorm = a.norm();

    Mat<T> J = Mat<T>::Zero(dimC, dimC);
    if (aNorm <= b) {
      J.setIdentity();
    } else if (aNorm <= -b) {
    } else {
      T aNormInv = T(1) / aNorm;
      auto aHat = a * aNormInv;

      T g = (b + aNorm) * aNormInv;
      T h = b * aNormInv;

      J.topLeftCorner(n, n) =
          T(0.5) * (g * Mat<T>::Identity(n, n) - h * (aHat * aHat.transpose()));

      J.block(0, n, n, 1) = T(0.5) * aHat;

      J.block(n, 0, 1, n) = T(0.5) * aHat.transpose();

      J(n, n) = T(0.5);
    }
    return J;
  }
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite

#endif
