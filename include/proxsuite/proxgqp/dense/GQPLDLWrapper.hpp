#ifndef PROXSUITE_GQP_LDL_WRAPPER
#define PROXSUITE_GQP_LDL_WRAPPER

#include "GQPWithInit.hpp"
#include "Strategy.hpp"
#include "fwd.hpp"
#include "proxsuite/linalg/dense/core.hpp"
#include "proxsuite/linalg/dense/ldlt.hpp"
#include "proxsuite/linalg/veg/memory/dynamic_stack.hpp"
#include "proxsuite/linalg/veg/vec.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/LU>
#include <iostream>
#include <ostream>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace proxgqp {
namespace dense {

template <typename T> struct StrategyState {
  Vec<T> prevSolution;
};

template <typename T> struct GQPLDLWrapper : BaseGQPWithInitSupported<T> {

  Timer kktConstructionInUpdateTimer;
  Timer solveUpdateTimer;

  linalg::dense::Ldlt<T> ldl;
  proxsuite::linalg::veg::Vec<unsigned char> ldl_stack;
  Mat<T> kkt;
  Vec<T> rhs;
  Vec<T> rhsExpanded;
  Eigen::PartialPivLU<Mat<T>> luSolver;
  Eigen::MINRES<Mat<T>> minresSolver;
  StrategyState<T> strategyState;

  GQPLDLWrapper(isize dim) : BaseGQPWithInitSupported<T>(dim) {}

  void _setupLDLT(isize kktDim, isize nDiag) {
    using namespace proxsuite::linalg::veg::dynstack;

    auto stackReq =
        ldl.factorize_req(kktDim) |
        (linalg::dense::temp_vec_req(proxsuite::linalg::veg::Tag<T>{}, nDiag) &
         StackReq::with_len(proxsuite::linalg::veg::Tag<isize>{}, nDiag) &
         ldl.diagonal_update_req(kktDim, nDiag)) |
        ldl.solve_in_place_req(kktDim);

    isize needed = stackReq.alloc_req();
    if (ldl_stack.len() < needed) {
      ldl.reserve_uninit(kktDim);
      ldl_stack.resize_for_overwrite(needed);
    }
  }

  auto _factorizeBase() {
    auto stack = _makeStack();
    ldl.factorize(kkt.transpose(), stack);
  }

  auto _makeStack() -> proxsuite::linalg::veg::dynstack::DynStackMut {
    return {proxsuite::linalg::veg::from_slice_mut, ldl_stack.as_mut()};
  }

  void _buildKKTBase() {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;
    isize kktDim = n + m_eq + m_in;

    if (kkt.rows() != kktDim) {
      kkt.resize(kktDim, kktDim);
      rhs.resize(kktDim);
    }

    this->_setupLDLT(kktDim, m_eq + m_in);

    kkt.setZero();
    kkt.topLeftCorner(n, n) = this->objectiveAggr.HScaled;
    kkt.topLeftCorner(n, n).diagonal().array() += this->settings.default_rho;

    isize off = n;
    for (auto const &eq : this->equalityConstraints) {
      isize mi = eq.AScaled.rows();
      kkt.block(0, off, n, mi) = eq.AScaled.transpose();
      kkt.block(off, 0, mi, n) = eq.AScaled;
      off += mi;
    }

    kkt.diagonal().segment(n, m_eq).setConstant(-this->settings.default_mu_eq);
    kkt.diagonal()
        .segment(n + m_eq, m_in)
        .setConstant(-this->settings.default_mu_in);

    _factorizeBase();
  }

  void _commonWithoutProductKKTConstruction() {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    isize kktDim = n + m_eq + 3 * m_in;

    if (kkt.rows() != kktDim) {
      kkt.resize(kktDim, kktDim);
      rhs.resize(n + m_eq + m_in);
      rhsExpanded.resize(kktDim);
    }

    kkt.setZero();
    kkt.topLeftCorner(n, n) = this->objectiveAggr.HScaled;
    kkt.topLeftCorner(n, n).diagonal().array() += this->settings.default_rho;

    isize off = n;
    for (auto const &eq : this->equalityConstraints) {
      isize mi = eq.AScaled.rows();
      kkt.block(0, off, n, mi) = eq.AScaled.transpose();
      kkt.block(off, 0, mi, n) = eq.AScaled;
      off += mi;
    }

    kkt.diagonal().segment(n, m_eq).setConstant(-this->settings.default_mu_eq);

    if (m_in > 0) {
      isize off_in = n + m_eq;
      isize off_beta = n + m_eq + m_in;
      isize off_alpha = n + m_eq + 2 * m_in;

      kkt.diagonal()
          .segment(off_in, m_in)
          .setConstant(-this->settings.default_mu_in);

      isize off = 0;
      for (auto const &ineq : this->inequalityConstraints) {
        isize dimC = ineq.dScaled.size();

        kkt.block(off_in + off, off_beta + off, dimC, dimC).setIdentity();
        kkt.block(off_beta + off, off_in + off, dimC, dimC).setIdentity();

        kkt.block(off_beta + off, off_alpha + off, dimC, dimC)
            .diagonal()
            .setConstant(T(-1));
        kkt.block(off_alpha + off, off_beta + off, dimC, dimC)
            .diagonal()
            .setConstant(T(-1));

        kkt.block(0, off_alpha + off, n, dimC) = ineq.CScaled.transpose();
        kkt.block(off_alpha + off, 0, dimC, n) = ineq.CScaled;

        off += dimC;
      }
    }
  }

  void _buildKKTBaseWithoutProduct() {
    _commonWithoutProductKKTConstruction();

    luSolver.compute(kkt);
  }

  void _buildKKTSimpleIterativeSolver() {
    _commonWithoutProductKKTConstruction();
  }

  void _buildKKT(GQPStrategy strategy = GQPStrategy::Base) {
    switch (strategy) {
    case GQPStrategy::Base: {
      _buildKKTBase();
      break;
    }
    case GQPStrategy::BaseWithoutProduct: {
      _buildKKTBaseWithoutProduct();
      break;
    }
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      _buildKKTSimpleIterativeSolver();
      break;
    }
    }
  }

  void _kktConstructionInUpdateBase(T muIn, VecRef<T> x, VecRef<T> zPrev) {
    isize n = this->dim;
    isize m_eq = this->n_eq;

    isize off = 0;
    for (auto const &ineq : this->inequalityConstraints) {
      isize dimC = ineq.dScaled.size();
      Vec<T> arg =
          muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
      auto J = ineq.cone.dualJacobian(arg);
      auto JC = J * ineq.CScaled;

      kkt.block(n + m_eq + off, 0, dimC, n) = JC;
      kkt.block(0, n + m_eq + off, n, dimC) = JC.transpose();
      off += dimC;
    }
  }

  void _solveUpdateKKTBase(T muIn, VecRef<T> x, VecRef<T> zPrev) {
    _factorizeBase();
  }

  void _commonInnerLoopKKTWithoutProductUpdate(T muIn, VecRef<T> x,
                                               VecRef<T> zPrev) {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize off_in = n + m_eq;
    isize off_beta = n + m_eq + this->n_in;

    isize off = 0;
    for (auto const &ineq : this->inequalityConstraints) {
      isize dimC = ineq.dScaled.size();
      Vec<T> arg =
          muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
      auto J = ineq.cone.dualJacobian(arg);

      kkt.block(off_in + off, off_beta + off, dimC, dimC) = J;
      kkt.block(off_beta + off, off_in + off, dimC, dimC) = J.transpose();

      off += dimC;
    }
  }

  void _updateKKTInInnerLoopBaseWithoutProduct(T muIn, VecRef<T> x,
                                               VecRef<T> zPrev) {
    _commonInnerLoopKKTWithoutProductUpdate(muIn, x, zPrev);
    luSolver.compute(kkt);
  }

  void _kktConstructionInUpdateWithoutProduct(T muIn, VecRef<T> x,
                                              VecRef<T> zPrev) {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize off_in = n + m_eq;
    isize off_beta = n + m_eq + this->n_in;

    isize off = 0;
    for (auto const &ineq : this->inequalityConstraints) {
      isize dimC = ineq.dScaled.size();
      Vec<T> arg =
          muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
      auto J = ineq.cone.dualJacobian(arg);

      kkt.block(off_in + off, off_beta + off, dimC, dimC) = J;
      kkt.block(off_beta + off, off_in + off, dimC, dimC) = J.transpose();

      off += dimC;
    }
  }

  void _constructionUpdate(T muIn, VecRef<T> x, VecRef<T> zPrev,
                           GQPStrategy strategy = GQPStrategy::Base) {
    switch (strategy) {
    case GQPStrategy::Base: {
      _kktConstructionInUpdateBase(muIn, x, zPrev);
      break;
    }
    case GQPStrategy::BaseWithoutProduct: {
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      _kktConstructionInUpdateWithoutProduct(muIn, x, zPrev);
      break;
    }
    }
    }
  }

  void _solveUpdateKKTBaseWithoutProduct() { luSolver.compute(kkt); }

  void _solveUpdate(T muIn, VecRef<T> x, VecRef<T> zPrev,
                    GQPStrategy strategy = GQPStrategy::Base) {
    switch (strategy) {
    case GQPStrategy::Base: {
      _solveUpdateKKTBase(muIn, x, zPrev);
      break;
    }
    case GQPStrategy::BaseWithoutProduct: {
      _solveUpdateKKTBaseWithoutProduct();
      break;
    }
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      // Iterative strategy don't need any refactorisation
      break;
    }
    }
  }

  virtual void _updateKKTInInnerLoop(T muIn, VecRef<T> x, VecRef<T> zPrev,
                                     GQPStrategy strategy = GQPStrategy::Base) {
    kktConstructionInUpdateTimer.start();
    _constructionUpdate(muIn, x, zPrev, strategy);
    kktConstructionInUpdateTimer.end();
    std::cout << "@KKT construction update time taken: "
              << kktConstructionInUpdateTimer.timeInMilliSeconds() << "ms"
              << std::endl;

    solveUpdateTimer.start();
    _solveUpdate(muIn, x, zPrev, strategy);
    solveUpdateTimer.end();
    std::cout << "@KKT solver update time taken: "
              << solveUpdateTimer.timeInMilliSeconds() << "ms" << std::endl;
  }

  void _updateBarrierParamsBase(T rho_old, T rho_new, T r_eq_old, T r_eq_new,
                                T r_in_old, T r_in_new) {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    auto stack = _makeStack();

    if (rho_old != rho_new) {
      kkt.topLeftCorner(n, n).diagonal().array() += rho_new - rho_old;
      ldl.factorize(kkt.transpose(), stack);
    }

    if (r_eq_old != r_eq_new || r_in_old != r_in_new) {
      isize r = m_eq + m_in;
      LDLT_TEMP_VEC_UNINIT(T, alpha, r, stack);
      alpha.head(m_eq).setConstant(r_eq_old - r_eq_new);
      alpha.tail(m_in).setConstant(r_in_old - r_in_new);

      auto indices_storage =
          stack.make_new_for_overwrite(proxsuite::linalg::veg::Tag<isize>{}, r);
      isize *indices = indices_storage.ptr_mut();
      for (isize k = 0; k < m_eq; ++k) {
        indices[k] = n + k;
      }
      for (isize k = 0; k < m_in; ++k) {
        indices[m_eq + k] = n + m_eq + k;
      }

      ldl.diagonal_update_clobber_indices(indices, r, alpha, stack);

      kkt.diagonal().segment(n, m_eq).setConstant(-r_eq_new);
      kkt.diagonal().segment(n + m_eq, m_in).setConstant(-r_in_new);
    }
  }

  void _commonUpdateBarrrierParamasWithoutProduct(T rho_old, T rho_new,
                                                  T r_eq_old, T r_eq_new,
                                                  T r_in_old, T r_in_new) {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    if (rho_old != rho_new) {
      kkt.topLeftCorner(n, n).diagonal().array() += rho_new - rho_old;
    }

    if (r_eq_old != r_eq_new) {
      kkt.diagonal().segment(n, m_eq).setConstant(-r_eq_new);
    }

    if (r_in_old != r_in_new) {
      kkt.diagonal().segment(n + m_eq, m_in).setConstant(-r_in_new);
    }
  }

  void _updateBarrierParamsBaseWithoutProduct(T rho_old, T rho_new, T r_eq_old,
                                              T r_eq_new, T r_in_old,
                                              T r_in_new) {

    _commonUpdateBarrrierParamasWithoutProduct(rho_old, rho_new, r_eq_old,
                                               r_eq_new, r_in_old, r_in_new);
    luSolver.compute(kkt);
  }

  void _updateBarrierParamsSimpleIterativeSolver(T rho_old, T rho_new,
                                                 T r_eq_old, T r_eq_new,
                                                 T r_in_old, T r_in_new) {
    _commonUpdateBarrrierParamasWithoutProduct(rho_old, rho_new, r_eq_old,
                                               r_eq_new, r_in_old, r_in_new);
  }

  void _updateBarrierParams(T rho_old, T rho_new, T r_eq_old, T r_eq_new,
                            T r_in_old, T r_in_new,
                            GQPStrategy strategy = GQPStrategy::Base) {

    switch (strategy) {
    case GQPStrategy::Base: {
      _updateBarrierParamsBase(rho_old, rho_new, r_eq_old, r_eq_new, r_in_old,
                               r_in_new);
      break;
    }
    case GQPStrategy::BaseWithoutProduct: {
      _updateBarrierParamsBaseWithoutProduct(rho_old, rho_new, r_eq_old,
                                             r_eq_new, r_in_old, r_in_new);
      break;
    }
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      _updateBarrierParamsSimpleIterativeSolver(rho_old, rho_new, r_eq_old,
                                                r_eq_new, r_in_old, r_in_new);
      break;
    }
    }
  }

  void _solveKKTBase(VecRefMut<T> rhs) {
    auto stack = _makeStack();
    ldl.solve_in_place(rhs, stack);
  }

  void _solveKKTWithoutProduct(VecRefMut<T> rhs, GQPStrategy strategy) {

    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    rhsExpanded.head(n + m_eq + m_in) = rhs;
    rhsExpanded.tail(2 * m_in).setZero();

    // std::cout << rhsExpanded << std::endl;

    switch (strategy) {
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      _solveKKTSimpleIterativeSolver(strategy);
      break;
    }
    case GQPStrategy::BaseWithoutProduct: {
      _solveKKTBaseWithoutProduct();
      break;
    }
    case GQPStrategy::Base: {
      // SHOULDN'T HAPPEN
      break;
    }
    }

    strategyState.prevSolution = rhsExpanded;

    rhs = rhsExpanded.head(n + m_eq + m_in);
  }

  void _solveKKTSimpleIterativeSolver(GQPStrategy strategy) {

    // std::cout << "KKT:" << kkt << std::endl;

    // printf("MINRES HERE\n");

    minresSolver.setMaxIterations(kkt.rows() * 2);
    minresSolver.setTolerance(this->settings.iterativeEpsilon);

    minresSolver.compute(kkt);

    Vec<T> rhsCopy = rhsExpanded;
    switch (strategy) {
    case GQPStrategy::SimpleIterativeSolverWithWarmStart: {
      rhsExpanded =
          minresSolver.solveWithGuess(rhsCopy, strategyState.prevSolution);
      break;
    }
    default: {
      rhsExpanded = minresSolver.solve(rhsCopy);
    }
    }

    // rhsExpanded =
    //    minresSolver.solve(rhsCopy, strategyState.prevSolution);

    /*
    std::cout << "=======" << std::endl;
    std::cout << "PREV:" << strategyState.prevSolution << std::endl;
    std::cout << "CUR:" << rhsExpanded << std::endl;
    std::cout << "______" << std::endl;
    */

    // std::cout << "#iterations:     " << minresSolver.iterations() <<
    // std::endl; std::cout << "estimated error: " << minresSolver.error() <<
    // std::endl;

    // std::cout << "AFTER MINRES:" << rhsExpanded << std::endl;
  }

  void _solveKKTBaseWithoutProduct() {
    // std::cout << "KKT:" << kkt << std::endl;
    // printf("LU HERE\n");
    Vec<T> rhsCopy = rhsExpanded;
    rhsExpanded = luSolver.solve(rhsCopy);

    // std::cout << "AFTER LU:" << rhsExpanded << std::endl;
  }

  void _solveKKT(VecRefMut<T> rhs, GQPStrategy strategy = GQPStrategy::Base) {
    // We use the same api for all solveKKT i.e we only take the rhs that
    // correspond to -(-rStatStar,-rEq,-rCone) and it should be overwritten
    // with dx,dy,dz , even if the system ahs more variable in fact like in
    // the Without product the extenriio on ly care about dx,dy,dz
    switch (strategy) {
    case GQPStrategy::Base: {
      _solveKKTBase(rhs);
      break;
    }
      // For strategy using the extended kkt without product
    case GQPStrategy::SimpleIterativeSolver:
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::BaseWithoutProduct: {
      _solveKKTWithoutProduct(rhs, strategy);
      break;
    }
    }
  }
};

} // namespace dense

} // namespace proxgqp
} // namespace proxsuite

#endif
