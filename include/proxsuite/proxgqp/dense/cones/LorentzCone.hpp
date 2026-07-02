#ifndef PROXSUITE_GQP_CONE_LORENTZ
#define PROXSUITE_GQP_CONE_LORENTZ

#include "proxsuite/proxgqp/dense/BaseGQP.hpp"
#include "proxsuite/proxgqp/dense/fwd.hpp"
#include <cmath>

namespace proxsuite {
namespace proxgqp {

template <typename T> struct LorentzCone : Cone<T> {
  isize dimC;

  LorentzCone(isize dim) : dimC(dim) {}

  Vec<T> applyJacobian(VecRef<T> arg, VecRef<T> x) override {
    return dualJacobian(arg) * x;
  }

  SparseMat<T> dualSparseJacobian(VecRef<T> z) override {
    return convertToSparseMat(dualJacobian(z));
  };

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

  Mat<T> fastMultByDualJacobian(VecRef<T> z, MatRef<T> C) override {
    isize n = dimC - 1;
    auto a = z.head(n);
    T b = z(n);
    T aNorm = a.norm();
    if (aNorm <= b) {
      return C;
    }
    if (aNorm <= -b) {
      return Mat<T>::Zero(dimC, C.cols());
    }
    T aNormInv = T(1) / aNorm;
    auto aHat = a * aNormInv;
    T g = (b + aNorm) * aNormInv;
    T h = b * aNormInv;
    auto C_top = C.topRows(n);
    auto aHatT_C = aHat.transpose() * C_top;
    Mat<T> JC(dimC, C.cols());
    JC.topRows(n) =
        T(0.5) * (g * C_top - h * aHat * aHatT_C) + T(0.5) * aHat * C.row(n);
    JC.row(n) = T(0.5) * aHatT_C + T(0.5) * C.row(n);
    return JC;
  }
  Mat<T> fastMultByPrecondJacobian(VecRef<T> z, MatRef<T> C) override {
    return fastMultByDualJacobian(z, C);
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

  Mat<T> order2Mat(VecRef<T> z, VecRef<T> u) {
    isize n = dimC - 1;
    auto a = z.head(n);
    T b = z(n);
    T aNorm = a.norm();

    Mat<T> M = Mat<T>::Zero(dimC, dimC);

    // In the flat regions, the Jacobian J(z) is constant (I or 0),
    // so its derivative with respect to z is 0.
    if (aNorm <= b || aNorm <= -b) {
      return M;
    }

    // Middle region derivative calculations
    T aNormInv = T(1) / aNorm;
    auto aHat = a * aNormInv;

    auto u1 = u.head(n);
    T u2 = u(n);

    // Projections of u1
    T c = aHat.dot(u1);
    Vec<T> w = u1 - c * aHat;

    T factor = T(0.5) * aNormInv;
    T coeff_P = u2 - b * c * aNormInv;
    T coeff_rank2 = b * aNormInv;

    // Top-Left Block (n x n)
    M.topLeftCorner(n, n) =
        factor * (coeff_P * Mat<T>::Identity(n, n) -
                  coeff_P * (aHat * aHat.transpose()) -
                  coeff_rank2 * (aHat * w.transpose() + w * aHat.transpose()));

    // Top-Right Block (n x 1)
    M.block(0, n, n, 1) = factor * w;

    // Bottom-Left Block (1 x n)
    M.block(n, 0, 1, n) = factor * w.transpose();

    // Bottom-Right Block (1 x 1) remains 0

    return M;
  }
};

} // namespace proxgqp
} // namespace proxsuite

#endif
