#include "proxsuite/proxsocp/dense/fwd.hpp"
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp> // used for generating a random convex qp
#include <proxsuite/proxsocp/dense/dense.hpp>
#include <vector>

using namespace proxsuite;
using T = double;

void test_classic_qp() {
  using namespace proxqp::dense;
  // generate a QP problem
  T sparsity_factor = 0.15;
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  T strong_convexity_factor(1.e-2);
  // we generate a qp, so the function used from helpers.hpp is
  // in proxqp namespace. The qp is in dense eigen format and
  // you can control its sparsity ratio and strong convexity factor.
  Model<T> qp_random = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);

  // load PROXQP solver with dense backend and solve the problem
  QP<T> qp(dim, n_eq, n_in);
  qp.init(qp_random.H, qp_random.g, qp_random.A, qp_random.b, qp_random.C,
          qp_random.l, qp_random.u);
  qp.solve();
  // print an optimal solution x,y and z
  std::cout << "optimal x: " << qp.results.x << std::endl;
  std::cout << "optimal y: " << qp.results.y << std::endl;
  std::cout << "optimal z: " << qp.results.z << std::endl;
}

void test_barebone_wrapper() {
  using namespace proxsocp::dense;
  isize dim(10);
  isize n_eq(2);
  isize n_in(2);
  std::vector<isize> dimInequality = {};

  SOCP<T> socp(dim, n_eq, n_in, dimInequality);

  Mat<T> H(dim, dim);
  H.setOnes();
}
