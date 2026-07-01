#ifndef PROXSUITE_BASE_GQP_WITHOUT_SOLVE
#define PROXSUITE_BASE_GQP_WITHOUT_SOLVE

#include "fwd.hpp"
#include "proxsuite/linalg/veg/internal/macros.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace proxsuite {
namespace proxgqp {

namespace dense {

template <typename T> struct InequalityConstraintAggregator {
  Mat<T> CScaled;
  Vec<T> dScaled;
  Mat<T> C;
  Vec<T> d;
  Cone<T> &cone;
  InequalityConstraintAggregator(MatRef<T> C, VecRef<T> d, Cone<T> &cone)
      : C(C), d(d), cone(cone), CScaled(C), dScaled(d) {}
};

template <typename T> struct EqualityConstraintAggregator {
  Mat<T> A;
  Vec<T> b;
  Mat<T> AScaled;
  Vec<T> bScaled;
  EqualityConstraintAggregator(MatRef<T> A, VecRef<T> b)
      : A(A), b(b), AScaled(A), bScaled(b) {}
};

template <typename T> struct ObjectifParamAggregator {
  Mat<T> H;
  Vec<T> g;
  Mat<T> HScaled;
  Vec<T> gScaled;
  ObjectifParamAggregator<T>(isize dim)
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

template <typename T> struct IdentityPreconditioner : GQPPreconditioner<T> {

  void adapt(vector<Eigen::Ref<Mat<T>>> &mats) {}
  void scaleRHS(Vec<T> &gScaled, vector<Eigen::Ref<Vec<T>>> &eqBScaled,
                vector<Eigen::Ref<Vec<T>>> &inDScaled) {};
  void scaleInPlace(Vec<T> &vec, isize startMatIndex, isize endMatIndex) {};
  void unscaleInPlace(Vec<T> &vec, isize startMatIndex, isize endMatIndex) {};
  void scaleCost(Vec<T> &vec) {};
  void scaleCost(Mat<T> &mat) {};
  void unscaleCost(Vec<T> &vec) {};
  void unscaleCost(Mat<T> &mat) {};
  void setParams(T eps, isize max_it) {}
};

template <typename T> struct RuizPreconditioner : GQPPreconditioner<T> {
  Vec<T> delta;
  T c = T(1);
  isize n_cols = 0;
  vector<isize> rowOffsets;
  vector<isize> rowSizes;
  T epsilon;
  isize max_iter;

  void setParams(T eps, isize max_it) override {
    epsilon = eps;
    max_iter = max_it;
  }

  void adapt(vector<Eigen::Ref<Mat<T>>> &mats) override {
    PROXSUITE_THROW_PRETTY(mats.empty(), std::invalid_argument,
                           "RuizPreconditioner::adapt: mats vector is empty");

    n_cols = mats[0].cols();
    for (isize i = 1; i < (isize)mats.size(); ++i) {
      PROXSUITE_THROW_PRETTY(mats[i].cols() != n_cols, std::invalid_argument,
                             "RuizPreconditioner::adapt: all matrices must "
                             "have the same number of columns");
    }

    isize const n_mats = mats.size();

    rowSizes.resize(n_mats);
    rowOffsets.resize(n_mats);

    isize total_rows = 0;
    for (isize k = 0; k < n_mats; ++k) {
      rowSizes[k] = mats[k].rows();
      total_rows += rowSizes[k];
    }

    isize offset = n_cols;
    for (isize k = 0; k < n_mats; ++k) {
      rowOffsets[k] = offset;
      offset += rowSizes[k];
    }

    static constexpr T machineEps = std::numeric_limits<T>::epsilon();
    isize const deltaSize = n_cols + total_rows;

    delta = Vec<T>::Ones(deltaSize);
    Vec<T> deltaLocal(deltaSize);

    isize iter = 1;
    while (true) {
      deltaLocal.setOnes();

      for (isize j = 0; j < n_cols; ++j) {
        T maxNorm = T(0);
        for (auto const &mat : mats) {
          if (mat.rows() > 0) {
            maxNorm = std::max(maxNorm,
                               mat.col(j).template lpNorm<Eigen::Infinity>());
          }
        }
        T aux = std::sqrt(maxNorm);
        deltaLocal(j) = (aux == T(0)) ? T(1) : T(1) / (aux + machineEps);
      }

      for (isize k = 0; k < n_mats; ++k) {
        auto &mat = mats[k];
        isize const roff = rowOffsets[k];
        for (isize r = 0; r < mat.rows(); ++r) {
          T aux = std::sqrt(mat.row(r).template lpNorm<Eigen::Infinity>());
          deltaLocal(roff + r) =
              (aux == T(0)) ? T(1) : T(1) / (aux + machineEps);
        }
      }

      for (isize k = 0; k < n_mats; ++k) {
        auto &mat = mats[k];
        mat = deltaLocal.segment(rowOffsets[k], mat.rows()).asDiagonal() * mat *
              deltaLocal.head(n_cols).asDiagonal();
      }

      delta.array() *= deltaLocal.array();

      if ((1 - deltaLocal.array())
              .matrix()
              .template lpNorm<Eigen::Infinity>() <= epsilon) {
        break;
      }
      if (iter == max_iter) {
        break;
      }
      ++iter;
    }

    auto &Href = mats[0];
    T sum = T(0);
    for (isize i = 0; i < Href.rows(); ++i) {
      sum += Href.row(i).template lpNorm<Eigen::Infinity>();
    }
    c = T(1) / std::max(T(1), sum / T(std::max(isize(1), Href.rows())));
    Href *= c;
  }

  void scaleRHS(Vec<T> &gScaled, vector<Eigen::Ref<Vec<T>>> &eqBScaled,
                vector<Eigen::Ref<Vec<T>>> &inDScaled) override {
    gScaled.array() *= delta.head(n_cols).array();
    isize k = 1;
    for (auto &b : eqBScaled) {
      b.array() *= delta.segment(rowOffsets[k], rowSizes[k]).array();
      ++k;
    }
    for (auto &d : inDScaled) {
      d.array() *= delta.segment(rowOffsets[k], rowSizes[k]).array();
      ++k;
    }
  }

  void scaleInPlace(Vec<T> &vec, isize start, isize end) override {
    if (start == 0) {
      vec.array() /= delta.head(n_cols).array();
      return;
    }
    isize vecOffset = 0;
    for (isize k = start; k < end; ++k) {
      vec.segment(vecOffset, rowSizes[k]).array() /=
          delta.segment(rowOffsets[k], rowSizes[k]).array();
      vecOffset += rowSizes[k];
    }
  }

  void unscaleInPlace(Vec<T> &vec, isize start, isize end) override {
    if (start == 0) {
      vec.array() *= delta.head(n_cols).array();
      return;
    }
    isize vecOffset = 0;
    for (isize k = start; k < end; ++k) {
      vec.segment(vecOffset, rowSizes[k]).array() *=
          delta.segment(rowOffsets[k], rowSizes[k]).array();
      vecOffset += rowSizes[k];
    }
  }

  void scaleCost(Vec<T> &vec) override { vec.array() *= c; }
  void scaleCost(Mat<T> &mat) override { mat *= c; }
  void unscaleCost(Vec<T> &vec) override { vec.array() /= c; }
  void unscaleCost(Mat<T> &mat) override { mat *= T(1) / c; }
};

template <typename T> struct SolutionState {
  Vec<T> x;
  Vec<T> y;
  Vec<T> z;
  Vec<T> xScaled;
  Vec<T> yScaled;
  Vec<T> zScaled;
  SolutionState(VecRef<T> x, VecRef<T> y, VecRef<T> z)
      : x(x), y(y), z(z), xScaled(x), yScaled(y), zScaled(z) {}
  SolutionState(isize dim) : x(dim), y(0), z(0) {}

  void setZero() {
    x.setZero();
    xScaled.setZero();
    y.setZero();
    yScaled.setZero();
    z.setZero();
    zScaled.setZero();
  }
};

template <typename T> struct BaseGQP {
  ObjectifParamAggregator<T> objectiveAggr;
  vector<InequalityConstraintAggregator<T>> inequalityConstraints;
  vector<EqualityConstraintAggregator<T>> equalityConstraints;
  bool autoNeedToReAdaptPreconditionement = false;
  isize dim;
  isize n_eq = 0;
  isize n_in = 0;
  RuizPreconditioner<T> defaultConditioner;
  GQPPreconditioner<T> *conditioner = nullptr;
  SolutionState<T> solutionState;
  bool debug = false;

  BaseGQP<T>(isize dim)
      : dim(dim), objectiveAggr(dim), inequalityConstraints(),
        equalityConstraints(), solutionState(dim) {
    conditioner = &defaultConditioner;
  }

  void setConditioner(GQPPreconditioner<T> &newConditioner) {
    conditioner = &newConditioner;
  }

  void setObjective(MatRef<T> H, VecRef<T> g) {
    objectiveAggr.H = H;
    objectiveAggr.g = g;
    autoNeedToReAdaptPreconditionement = true;
  }

  void addEqualityConstraint(MatRef<T> A, VecRef<T> b) {
    EqualityConstraintAggregator<T> newEqualityConstraint(A, b);
    equalityConstraints.push_back(newEqualityConstraint);

    n_eq += b.size();

    autoNeedToReAdaptPreconditionement = true;

    solutionState.y.resize(solutionState.y.size() + b.size());
    solutionState.yScaled.resize(solutionState.y.size() + b.size());
  }

  void addInequalityConstraint(MatRef<T> C, VecRef<T> d, Cone<T> &cone) {
    InequalityConstraintAggregator<T> newInequalityConstraint(C, d, cone);
    inequalityConstraints.push_back(newInequalityConstraint);

    n_in += d.size();

    autoNeedToReAdaptPreconditionement = true;

    solutionState.z.resize(solutionState.z.size() + d.size());
    solutionState.zScaled.resize(solutionState.z.size() + d.size());
  }

  virtual void _readaptPreconditionementIfNeeded() {
    if (!autoNeedToReAdaptPreconditionement) {
      return;
    }

    _readaptPreconditionement();
    autoNeedToReAdaptPreconditionement = false;
  }

  void _readaptPreconditionement() {
    vector<Eigen::Ref<Mat<T>>> mats;

    objectiveAggr.HScaled = objectiveAggr.H;
    objectiveAggr.gScaled = objectiveAggr.g;

    mats.push_back(objectiveAggr.HScaled);

    vector<Eigen::Ref<Vec<T>>> eqBScaled;
    vector<Eigen::Ref<Vec<T>>> inDScaled;

    for (EqualityConstraintAggregator<T> &equalityAggr : equalityConstraints) {
      equalityAggr.AScaled = equalityAggr.A;
      equalityAggr.bScaled = equalityAggr.b;
      eqBScaled.push_back(equalityAggr.bScaled);
      mats.push_back(equalityAggr.AScaled);
    }

    for (InequalityConstraintAggregator<T> &inequalityAggr :
         inequalityConstraints) {
      inequalityAggr.CScaled = inequalityAggr.C;
      inequalityAggr.dScaled = inequalityAggr.d;
      inDScaled.push_back(inequalityAggr.dScaled);
      mats.push_back(inequalityAggr.CScaled);
    }

    conditioner->adapt(mats);
    conditioner->scaleRHS(objectiveAggr.gScaled, eqBScaled, inDScaled);
    conditioner->scaleCost(objectiveAggr.gScaled);
  }

  void _scaleSolution() {
    _readaptPreconditionementIfNeeded();
    isize startMatIndex{0};
    isize endMatIndex{1};

    solutionState.xScaled = solutionState.x;
    conditioner->scaleInPlace(solutionState.xScaled, startMatIndex,
                              endMatIndex);

    startMatIndex = endMatIndex;
    endMatIndex += equalityConstraints.size();

    solutionState.yScaled = solutionState.y;
    conditioner->scaleInPlace(solutionState.yScaled, startMatIndex,
                              endMatIndex);
    conditioner->scaleCost(solutionState.yScaled);

    startMatIndex = endMatIndex;
    endMatIndex += inequalityConstraints.size();

    solutionState.zScaled = solutionState.z;
    conditioner->scaleInPlace(solutionState.zScaled, startMatIndex,
                              endMatIndex);
    conditioner->scaleCost(solutionState.zScaled);
  }

  void _unscaleSolution() {
    _readaptPreconditionementIfNeeded();

    isize startMatIndex{0};
    isize endMatIndex{1};

    solutionState.x = solutionState.xScaled;
    conditioner->unscaleInPlace(solutionState.x, startMatIndex, endMatIndex);

    startMatIndex = endMatIndex;
    endMatIndex += equalityConstraints.size();

    solutionState.y = solutionState.yScaled;
    conditioner->unscaleCost(solutionState.y);
    conditioner->unscaleInPlace(solutionState.y, startMatIndex, endMatIndex);

    startMatIndex = endMatIndex;
    endMatIndex += inequalityConstraints.size();

    solutionState.z = solutionState.zScaled;
    conditioner->unscaleCost(solutionState.z);
    conditioner->unscaleInPlace(solutionState.z, startMatIndex, endMatIndex);
  }
};

} // namespace dense

} // namespace proxgqp
} // namespace proxsuite

#endif
