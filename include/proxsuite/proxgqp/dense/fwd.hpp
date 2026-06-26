#ifndef PROXSUITE_PROXSOCP_DENSE_FWD_HPP
#define PROXSUITE_PROXSOCP_DENSE_FWD_HPP

#include "proxsuite/helpers/common.hpp"
#include "proxsuite/linalg/veg/internal/typedefs.hpp"
#include "proxsuite/proxgqp/dense/settings.hpp"
#include "proxsuite/proxqp/results.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <chrono>

namespace proxsuite {
namespace proxgqp {

static constexpr auto DYN = Eigen::Dynamic;
enum { layout = Eigen::RowMajor };
template <typename T> using SparseMat = Eigen::SparseMatrix<T, 1>;

template <typename T> using Vec = Eigen::Matrix<T, DYN, 1>;
template <typename T> using VecRef = Eigen::Ref<Vec<T> const>;
template <typename T> using VecRefMut = Eigen::Ref<Vec<T>>;

template <typename T, int l = layout> using Mat = Eigen::Matrix<T, DYN, DYN, l>;

template <typename T, int l = layout>
using MatRef = Eigen::Ref<Mat<T, l> const>;

using isize = proxsuite::linalg::veg::isize;

template <typename T> using VecMap = Eigen::Map<Vec<T> const>;
template <typename T> using VecMapMut = Eigen::Map<Vec<T>>;

template <typename T, int l = layout>
using MatMap = Eigen::Map<Mat<T, l> const>;
template <typename T, int l = layout> using MatMapMut = Eigen::Map<Mat<T, l>>;

using VecMapISize = Eigen::Map<Eigen::Matrix<isize, DYN, 1> const>;
using VecISize = Eigen::Matrix<isize, DYN, 1>;

using VecMapBool = Eigen::Map<Eigen::Matrix<bool, DYN, 1> const>;
using VecBool = Eigen::Matrix<bool, DYN, 1>;
template <typename T> using Results = proxsuite::proxqp::Results<T>;
template <typename T> struct Cone {
  virtual Mat<T> dualJacobian(VecRef<T> x) = 0;
  virtual SparseMat<T> dualSparseJacobian(VecRef<T> x) = 0;
  virtual Vec<T> applyJacobian(VecRef<T> arg, VecRef<T> x) = 0;
  virtual Vec<T> dualProject(VecRef<T> x) = 0;
};

template <typename T> using SparseMatRef = Eigen::Ref<SparseMat<T>>;

template <typename T> SparseMat<T> convertToSparseMat(Mat<T> M) {
  typedef Eigen::Triplet<T> Triple;
  std::vector<Triple> tripletList;
  for (isize i = 0; i < M.rows(); i++) {
    for (isize j = 0; j < M.cols(); j++) {
      if (M(i, j) != 0) {
        tripletList.push_back(Triple(i, j, M(i, j)));
      }
    }
  }
  SparseMat<T> out(M.rows(), M.cols());

  out.setFromTriplets(tripletList.begin(), tripletList.end());
  return out;
}

struct Timer {
  long startTime = 0;
  long endTime = 0;
  double accumulatedTime = 0;

  void reset() {
    this->startTime = 0;
    this->endTime = 0;
  }
  double accumulated() { return accumulatedTime; }

  void start() {
    startTime =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }

  void end() {
    endTime =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    accumulatedTime += timeInMilliSeconds();
  }

  double timeInMilliSeconds() { return (endTime - startTime) / 1e6; }
};

} // namespace proxgqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXQP_DENSE_FWD_HPP */
