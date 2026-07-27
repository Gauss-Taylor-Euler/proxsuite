#ifndef PROXSUITE_BASE_GQP_WITH_INIT_Supported
#define PROXSUITE_BASE_GQP_WITH_INIT_Supported

#include "BaseGQP.hpp"
#include "fwd.hpp"
#include <Eigen/Cholesky>

namespace proxsuite {
namespace proxgqp {
namespace dense {

template <typename T> struct BaseGQPWithInitSupported : BaseGQP<T> {
  bool exteriorInited = false;
  GQPSettings<T> settings;

  BaseGQPWithInitSupported(isize dim) : BaseGQP<T>(dim) {}

  void _readaptPreconditionementIfNeeded() override {
    this->conditioner->setParams(settings.preconditioner_accuracy,
                                 settings.preconditioner_max_iter);
    BaseGQP<T>::_readaptPreconditionementIfNeeded();
  }

  void _initSolutionWithZero() { this->solutionState.setZero(); }

  void initSolutionWithZero() {
    this->_initSolutionWithZero();
    exteriorInited = true;
  }
  void _initSolutionWithEqualitySolution() {
    initSolutionWithZero();
    this->_readaptPreconditionement();

    isize n = this->dim;
    isize m = this->n_eq;

    Mat<T> kktEqualityOnly(n + m, n + m);

    kktEqualityOnly.topLeftCorner(n, n) = this->objectiveAggr.HScaled;
    kktEqualityOnly.topLeftCorner(n, n).diagonal().array() +=
        settings.default_rho;

    isize offset = 0;
    for (auto const &eq : this->equalityConstraints) {
      isize mi = eq.AScaled.rows();
      kktEqualityOnly.block(0, n + offset, n, mi) = eq.AScaled.transpose();
      kktEqualityOnly.block(n + offset, 0, mi, n) = eq.AScaled;
      offset += mi;
    }

    kktEqualityOnly.bottomRightCorner(m, m).setZero();
    kktEqualityOnly.diagonal().segment(n, m).setConstant(
        -settings.default_mu_eq);

    Vec<T> rhs(n + m);
    rhs.head(n) = -this->objectiveAggr.gScaled;
    offset = 0;
    for (auto const &eq : this->equalityConstraints) {
      rhs.segment(n + offset, eq.bScaled.size()) = eq.bScaled;
      offset += eq.bScaled.size();
    }

    Eigen::LDLT<Mat<T>> solver;
    solver.compute(kktEqualityOnly);
    Vec<T> sol = solver.solve(rhs);

    this->solutionState.xScaled = sol.head(n);
    this->solutionState.yScaled = sol.tail(m);

    this->_unscaleSolution();
  }

  void initSolutionWithEqualitySolution() {
    _initSolutionWithEqualitySolution();
    exteriorInited = true;
  }

  void initSolutionWithPreviousResult() {
    this->_scaleSolution();
    exteriorInited = true;
  }

  void initSolutionWithWarmStart(VecRef<T> x, VecRef<T> y, VecRef<T> z) {
    this->solutionState = SolutionState<T>(x, y, z);
    this->_scaleSolution();
    exteriorInited = true;
  }
};

} // namespace dense

} // namespace proxgqp
} // namespace proxsuite

#endif
