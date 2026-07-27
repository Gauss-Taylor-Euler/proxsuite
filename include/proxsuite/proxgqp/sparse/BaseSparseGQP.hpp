#ifndef PROXSUITE_BASE_SPARSE_GQP_WITHOUT_SOLVE
#define PROXSUITE_BASE_SPARSE_GQP_WITHOUT_SOLVE
#include "fwd.hpp"
#include "proxsuite/proxgqp/dense/fwd.hpp"

namespace proxsuite {
namespace proxgqp {
namespace sparse {
template <typename T> struct SparseEqualityConstraintAggregator {
  SparseMat<T> A;
  Vec<T> b;
  Mat<T> AScaled;
  Vec<T> bScaled;
  SparseEqualityConstraintAggregator(SparseMatRef<T> A, VecRef<T> b)
      : A(A), b(b), AScaled(A), bScaled(b) {}
};
template <typename T> struct SparseInequalityConstraintAggregator {
  SparseMat<T> CScaled;
  Vec<T> dScaled;
  SparseMat<T> C;
  Vec<T> d;
  Cone<T> &cone;
  SparseInequalityConstraintAggregator(SparseMatRef<T> C, VecRef<T> d,
                                       Cone<T> &cone)
      : C(C), d(d), cone(cone), CScaled(C), dScaled(d) {}
};
template <typename T> struct SparseObjectifParamAggregator {
  SparseMat<T> H;
  Vec<T> g;
  SparseMat<T> HScaled;
  Vec<T> gScaled;
  SparseObjectifParamAggregator<T>(isize dim)
      : H(dim, dim), g(dim), HScaled(dim, dim), gScaled(dim) {}
};

using std::vector;
template <typename T> struct GQPPreconditioner {
  virtual void adapt(vector<Eigen::Ref<Mat<T>>> &mats) = 0;
  virtual void scaleRHS(Vec<T> &gScaled, vector<Eigen::Ref<Vec<T>>> &eqBScaled,
                        vector<Eigen::Ref<Vec<T>>> &inDScaled) = 0;
  virtual void scaleInPlace(Vec<T> &vec, isize startMatIndex,
                            isize endMatIndex) = 0;
  virtual void unscaleInPlace(Vec<T> &vec, isize startMatIndex,
                              isize endMatIndex) = 0;
  virtual void scaleCost(Vec<T> &vec) = 0;
  virtual void scaleCost(Mat<T> &mat) = 0;
  virtual void unscaleCost(Vec<T> &vec) = 0;
  virtual void unscaleCost(Mat<T> &mat) = 0;
  virtual void setParams(T eps, isize max_it) {}
};

template <typename T> struct BaseSparseGQP {
  BaseSparseGQP<T>(isize dim){};
};
} // namespace sparse
} // namespace proxgqp

} // namespace proxsuite

#endif
