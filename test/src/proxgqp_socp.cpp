#include <Eigen/Core>
#include <cmath>
#include <doctest.hpp>
#include <iostream>
#include <proxsuite/proxgqp/dense/cones/LorentzCone.hpp>
#include <proxsuite/proxgqp/dense/cones/PositiveOrthantCone.hpp>
#include <proxsuite/proxgqp/dense/dense.hpp>

using T = double;
using namespace proxsuite::proxgqp::dense;

DOCTEST_TEST_CASE(
    "proxgqp simple test: Lorentz cone + equality + orthant, 3 variables") {

  T epsAbs = T(1e-3);
  isize dim = 3;

  GQP<T> solver(dim);
  solver.settings.eps_abs = epsAbs;

  // objective: min -x_1 - 2x_1 + 0.5x_3   (H=0)
  Mat<T> H = Mat<T>::Zero(dim, dim);
  Vec<T> g(dim);
  g << T(-1), T(-2), T(0.5);
  solver.setObjective(H, g);

  // equality: x_1 + x_2 = 1
  Mat<T> A(1, dim);
  A << T(1), T(1), T(0);
  Vec<T> b(1);
  b << T(1);
  solver.addEqualityConstraint(A, b);

  // Lorentz cone L^4:   ||x||_2 ≤ 2  \iff  [-x_1, -x_2, -x_3, -2] \leq_{L^4} 0
  Mat<T> CLor(4, dim);
  CLor << T(-1), T(0), T(0), T(0), T(-1), T(0), //
      T(0), T(0), T(-1),                        //
      T(0), T(0), T(0);
  Vec<T> dLor(4);
  dLor << T(0), T(0), T(0), T(-2);
  LorentzCone<T> coneLor(4);
  solver.addInequalityConstraint(CLor, dLor, coneLor);

  // Orthant cone R^1_+:   x_3 ≥ 0  \iff  -x_3 ≤ 0
  Mat<T> COrt(1, dim);
  COrt << T(0), T(0), T(-1);
  Vec<T> dOrt(1);
  dOrt << T(0);
  PositiveOrthantCone<T> coneOrt(1);
  solver.addInequalityConstraint(COrt, dOrt, coneOrt);

  // --- solve ---
  GQPResult<T> result = solver.solve(true);

  DOCTEST_CHECK(result.status == GQPSolverStatus::GQP_SOLVED);
  DOCTEST_CHECK(result.pri_res <= epsAbs);
  DOCTEST_CHECK(result.dua_res <= epsAbs);
  DOCTEST_CHECK(result.outerIters >= 0);
  DOCTEST_CHECK(result.totalInnerIters >= 0);

  std::cout << "x = " << result.x.transpose() << std::endl;
  std::cout << "y = " << result.y.transpose() << std::endl;
  std::cout << "z = " << result.z.transpose() << std::endl;
  std::cout << "pri_res=" << result.pri_res << "  dua_res=" << result.dua_res
            << "  mu_eq=" << result.mu_eq << "  mu_in=" << result.mu_in
            << "  rho=" << result.rho << std::endl;
  std::cout << "outerIters=" << result.outerIters
            << "  totalInnerIters=" << result.totalInnerIters << std::endl;
}
