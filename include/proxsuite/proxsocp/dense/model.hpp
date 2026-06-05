#include "fwd.hpp"
#include <Eigen/Core>
#include <vector>
// #include "proxsuite/linalg/veg/internal/typedefs.hpp"
#include "proxsuite/helpers/optional.hpp"
#include "proxsuite/linalg/veg/internal/macros.hpp"
#include "proxsuite/linalg/veg/internal/typedefs.hpp"
#include "proxsuite/linalg/veg/type_traits/core.hpp"
namespace proxsuite {
namespace proxsocp {
namespace dense {
template <typename T> struct Model {
  // We add here H even through only g is used for socp(i.e H=0)
  // Mainly to keep more intersection with proxqp and also cause
  // The algorithm work even with H\neq 0
  // Here the objective is :
  // 1/2 <Hx,x>+<g,x>
  Mat<T> H;
  Vec<T> g;
  Mat<T> A;
  Vec<T> b;
  isize dim;
  isize n_eq;
  isize n_in;
  isize n_total;
  // In ||A_ix+b_i||\leq c_i^T x+d_i
  // dInequality[i] <-> d_i
  // cInequality[i] <-> c_i
  // bInequality[i] <-> b_i
  // AInequality[i] <-> A_i
  Vec<T> dInequality;
  Mat<T> cInequality;
  std::vector<Mat<T>> AInequality;
  // We use here a vector instead of a Mat
  // Because each bIn can have a different second dimension
  std::vector<Vec<T>> bInequality;
  // dimInequality[i] -> Give us the number of line of AInequality[i] and
  // bInequality[i]
  std::vector<isize> dimInequality;

  /*!
   * Default constructor.
   * @param dim primal variable dimension.
   * @param n_eq number of equality constraint
   * @param n_in number of inequality constraint
   * @param dimIn vector of first dimension of each A_i,b_i
   */
  Model(isize dim, isize n_eq, isize n_in, std::vector<isize> dimIn)
      : dim(dim), n_eq(n_eq), n_in(n_in), n_total(n_in + n_eq),
        dimInequality(dimIn), g(dim), A(n_eq, dim), b(n_eq),
        dInequality(n_in), cInequality(n_in, dim), AInequality(), bInequality(),
        H(dim, dim) {
    PROXSUITE_THROW_PRETTY(dim == 0, std::invalid_argument,
                           "wrong argument size: the dimension wrt the primal "
                           "variable x should be strictly positive.");

    g.setZero();
    A.setZero();
    b.setZero();
    dInequality.setZero();
    cInequality.setZero();
    H.setZero();

    for (isize i = 0; i < n_in; i++) {
      bInequality.push_back(Vec<T>(dimIn[i]));
      AInequality.push_back(Mat<T>(dimIn[i], dim));

      AInequality[i].setZero();
      bInequality[i].setZero();
    }
  }

  void setAndCheck(optional<MatRef<T>> &_H, optional<VecRef<T>> &_g,
                   optional<MatRef<T>> &_A, optional<VecRef<T>> &_b,
                   Vec<T> &_dInequality, Mat<T> &_cInequality,
                   std::vector<Mat<T>> &_AInequality,
                   std::vector<Vec<T>> &_bInequality) {

    if (_H != nullopt) {
      H = *_H;
    }

    if (_g != nullopt) {
      g = *_g;
    }

    if (_A != nullopt) {
      A = *_A;
    }

    if (_b != nullopt) {
      b = *_b;
    }

    dInequality = _dInequality;

    cInequality = _cInequality;

    PROXSUITE_CHECK_ARGUMENT_SIZE(
        _bInequality.size(), n_in,
        "New bInequality has not the expected format.");

    PROXSUITE_CHECK_ARGUMENT_SIZE(
        _AInequality.size(), n_in,
        "New AInequality has not the expected format.");

    for (isize k = 0; k < n_in; k++) {

      Vec<T> newB(_bInequality[k].size());

      for (isize i = 0; i < _bInequality[k].size(); i++) {
        newB[i] = _bInequality[k][i];
      }

      Mat<T> newA(_AInequality[k].rows(), _AInequality[k].cols());

      for (isize i = 0; i < _AInequality[k].rows(); i++) {
        for (isize j = 0; j < _AInequality[k].cols(); j++) {
          newA(i, j) = _AInequality[k](i, j);
        }
      }

      bInequality[k] = newB;
      AInequality[k] = newA;
    }

    assert(is_valid());
  }

  bool is_valid() {
    PROXSUITE_CHECK_ARGUMENT_SIZE(g.size(), dim, "g has not the expected size.")
    PROXSUITE_CHECK_ARGUMENT_SIZE(b.size(), n_eq,
                                  "b has not the expected size.")
    PROXSUITE_CHECK_ARGUMENT_SIZE(dInequality.size(), n_in,
                                  "dInequality has not the expected size");
    PROXSUITE_CHECK_ARGUMENT_SIZE(dimInequality.size(), n_in,
                                  "dimInequality has not the expected size.");

    if (cInequality.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
          cInequality.rows(), n_in,
          "cInequality has not the expected number of rows.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(
          cInequality.cols(), dim,
          "cInequality has not the expected number of cols.");
    }

    if (bInequality.size() || AInequality.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(
          bInequality.size(), n_in, "bInequality has not the expected format.");

      PROXSUITE_CHECK_ARGUMENT_SIZE(
          AInequality.size(), n_in, "AInequality has not the expected format.");

      for (isize k = 0; k < n_in; k++) {
        PROXSUITE_CHECK_ARGUMENT_SIZE(
            bInequality[k].size(), dimInequality[k],
            "bInequality has not the expected format.");
        PROXSUITE_CHECK_ARGUMENT_SIZE(
            AInequality[k].rows(), dimInequality[k],
            "AInequality has not the expected format.");
        PROXSUITE_CHECK_ARGUMENT_SIZE(
            AInequality[k].cols(), dim,
            "AInequality has not the expected format.");
      }
    }

    if (H.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(H.rows(), dim,
                                    "H has not the expected number of rows.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(H.cols(), dim,
                                    "H has not the expected number of cols.");
      PROXSUITE_THROW_PRETTY(
          (!H.isApprox(
              H.transpose(),
              std::numeric_limits<typename decltype(H)::Scalar>::epsilon())),
          std::invalid_argument, "H is not symmetric.");
    }
    if (A.size()) {
      PROXSUITE_CHECK_ARGUMENT_SIZE(A.rows(), n_eq,
                                    "A has not the expected number of rows.");
      PROXSUITE_CHECK_ARGUMENT_SIZE(A.cols(), dim,
                                    "A has not the expected number of cols.");
    }
  }
};
template <typename T>
bool operator==(const Model<T> &model1, const Model<T> &model2) {
  bool value = model1.dim == model2.dim && model1.n_eq == model2.n_eq &&
               model1.n_in == model2.n_in && model1.n_total == model2.n_total &&
               model1.g == model2.g && model1.A == model2.A &&
               model1.b == model2.b &&
               model1.cInequality == model2.cInequality &&
               model1.dInequality == model2.dInequality;

  isize n_in = model1.n_in;

  for (isize k = 0; k < n_in; k++) {
    if (!value) {
      break;
    }
    if (model1.AInequality[k] != model2.AInequality[k]) {
      return false;
    }
    if (model1.bInequality[k] != model2.bInequality[k]) {
      return false;
    }
  }
  return value;
}

template <typename T>
bool operator!=(const Model<T> &model1, const Model<T> &model2) {
  return !(model1 == model2);
}
} // namespace dense
} // namespace proxsocp

} // namespace proxsuite
