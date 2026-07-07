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
#include <Eigen/SparseCholesky>
#include <iostream>
#include <ostream>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace proxgqp {
namespace dense {
template <typename T> struct LuPrecondWrapper {
  Eigen::PartialPivLU<Mat<T>> const *lu = nullptr;

  using Index = isize;
  Index rows() const { return lu->rows(); }
  Index cols() const { return lu->cols(); }

  template <typename Rhs>
  auto solve(const Eigen::MatrixBase<Rhs> &b) const
      -> decltype(lu->solve(b.derived())) {
    return lu->solve(b.derived());
  }

  template <typename MatType> void compute(const MatType &) {}

  Eigen::ComputationInfo info() const {
    return Eigen::ComputationInfo::Success;
  }
};
template <typename T> struct LdltPrecondWrapper {
  linalg::dense::Ldlt<T> *ldlt = nullptr;
  proxsuite::linalg::veg::Vec<unsigned char> *stack_buf = nullptr;

  using Index = isize;
  Index rows() const { return ldlt->rows(); }
  Index cols() const { return ldlt->cols(); }

  Vec<T> solve(const Eigen::MatrixBase<Vec<T>> &b) const {
    Vec<T> x = b;
    proxsuite::linalg::veg::dynstack::DynStackMut stack{
        proxsuite::linalg::veg::from_slice_mut, stack_buf->as_mut()};
    ldlt->solve_in_place(x, stack);
    return x;
  }

  template <typename MatType> void compute(const MatType &) {}

  Eigen::ComputationInfo info() const {
    return Eigen::ComputationInfo::Success;
  }
};

template <typename T> struct StrategyState {
  Vec<T> prevSolution;
  T lastStep;
  isize currentOuter;
  Vec<T> lastXFactorise;
};

template <typename T> struct GQPLDLWrapper : BaseGQPWithInitSupported<T> {
  Vec<T> xIterate, yIterate, zIterate;
  Vec<T> xPrevOuter, yPrevOuter, zPrevOuter;

  Vec<T> rStat, rStatStar, rEq, rCone, rDMw;

  T muIn, muEq, rho;

  bool alwaysDL = true;
  bool ignoreCurvature = false;
  Timer kktConstructionInUpdateTimer;
  Timer solveUpdateTimer;

  linalg::dense::Ldlt<T> ldl;
  proxsuite::linalg::veg::Vec<unsigned char> ldl_stack;
  Mat<T> oldCurvature;
  Mat<T> kkt;
  SparseMat<T> sparseKKT;
  Mat<T> staleKKT;
  Vec<T> rhs;
  Vec<T> rhsExpanded;
  Eigen::PartialPivLU<Mat<T>> luSolver;
  Eigen::MINRES<Mat<T>> minresSolver;
  Eigen::GMRES<Mat<T>> gminresSolver;
  Eigen::GMRES<Mat<T>, LuPrecondWrapper<T>> precondGmresSolver;
  StrategyState<T> strategyState;
  LuPrecondWrapper<T> precondWrapper;

  linalg::dense::Ldlt<T> precondLdlt;
  proxsuite::linalg::veg::Vec<unsigned char> precondLdltStack;
  LdltPrecondWrapper<T> ldltPrecondWrapper;
  Eigen::GMRES<Mat<T>, LdltPrecondWrapper<T>> gmresLdltSolver;

  Eigen::SimplicialLDLT<SparseMat<T>> sparseLdLt;

  GQPLDLWrapper(isize dim)
      : BaseGQPWithInitSupported<T>(dim), xIterate(dim), yIterate(0),
        zIterate(0), xPrevOuter(dim), yPrevOuter(0), zPrevOuter(0) {}

  Mat<T> curvature() {
    Mat<T> out = Mat<T>::Zero(this->dim, this->dim);
    if (!this->ignoreCurvature) {
      std::cout << "\n###CURVATURE USED###" << std::endl;
      isize off = 0;
      for (auto const &ineq : this->inequalityConstraints) {
        isize dimC = ineq.dScaled.size();

        Vec<T> arg = muIn * this->zPrevOuter.segment(off, dimC) +
                     ineq.CScaled * xIterate + ineq.dScaled;

        // std::cout << "zIterate=" << zIterate << std::endl;
        // std::cout << "rCone=" << rCone << std::endl;

        Vec<T> u = zIterate.segment(off, dimC) -
                   (T(2) / muIn) * rCone.segment(off, dimC);

        Mat<T> M = ineq.cone.order2Mat(arg, u);

        out += ineq.CScaled.transpose() * M * ineq.CScaled;

        off += dimC;
      }
    }
    oldCurvature = out;
    return out;
  }

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

  void ldlRankUpdate(MatRef<T> w, VecRef<T> alpha) {
    isize n = ldl.dim();
    isize r = w.cols();

    using proxsuite::linalg::dense::temp_mat_req;

    auto copy_req = temp_mat_req(proxsuite::linalg::veg::Tag<T>{}, n, r);
    auto update_req = ldl.rank_r_update_req(n, r);
    auto total_req = copy_req & update_req;

    isize needed = total_req.alloc_req();
    if (ldl_stack.len() < needed) {
      ldl_stack.resize_for_overwrite(needed);
    }

    auto stack = _makeStack();
    LDLT_TEMP_MAT_UNINIT(T, w_col, n, r, stack);
    w_col = w;

    ldl.rank_r_update(w_col, alpha, stack);
  }

  void _commonWithProductConstruction() {
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
    kkt.topLeftCorner(n, n) = this->objectiveAggr.HScaled + curvature();
    kkt.topLeftCorner(n, n).diagonal().array() += rho;

    isize off = n;
    for (auto const &eq : this->equalityConstraints) {
      isize mi = eq.AScaled.rows();
      kkt.block(0, off, n, mi) = eq.AScaled.transpose();
      kkt.block(off, 0, mi, n) = eq.AScaled;
      off += mi;
    }

    kkt.diagonal().segment(n, m_eq).setConstant(-muEq);
    kkt.diagonal().segment(n + m_eq, m_in).setConstant(-muIn);

    _commonKKTConstructionWithProductUpdate(this->settings.mu_min_in, xIterate,
                                            zIterate);
  }

  void _buildKKTBase() {
    _commonWithProductConstruction();
    _factorizeBase();

    this->strategyState.lastXFactorise = xIterate;
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
    kkt.topLeftCorner(n, n) = this->objectiveAggr.HScaled + curvature();
    kkt.topLeftCorner(n, n).diagonal().array() += rho;

    isize off = n;
    for (auto const &eq : this->equalityConstraints) {
      isize mi = eq.AScaled.rows();
      kkt.block(0, off, n, mi) = eq.AScaled.transpose();
      kkt.block(off, 0, mi, n) = eq.AScaled;
      off += mi;
    }

    kkt.diagonal().segment(n, m_eq).setConstant(-muEq);

    if (m_in > 0) {
      isize off_in = n + m_eq;
      isize off_beta = n + m_eq + m_in;
      isize off_alpha = n + m_eq + 2 * m_in;

      kkt.diagonal().segment(off_in, m_in).setConstant(-muIn);

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
    _commonInnerLoopKKTWithoutProductUpdate(this->settings.mu_min_in, xIterate,
                                            zIterate);
  }

  void _buildKKTBaseWithoutProduct() {
    _commonWithoutProductKKTConstruction();

    luSolver.compute(kkt);
  }

  void _buildKKTSimpleIterativeSolver() {
    _commonWithoutProductKKTConstruction();
  }

  void _buildKKTBaseSparseLDLT() {
    _commonWithProductConstruction();
    _factorizeSparseLDLT();
  }

  void _buildKKT(GQPStrategy strategy = GQPStrategy::Base) {
    switch (strategy) {
    case GQPStrategy::BaseSparseLDLT: {
      _buildKKTBaseSparseLDLT();
      break;
    }
    case GQPStrategy::BaseProxqpLike:
    case GQPStrategy::Base: {
      _buildKKTBase();
      break;
    }
    case GQPStrategy::DL:
    case GQPStrategy::BaseWithoutProduct: {
      _buildKKTBaseWithoutProduct();
      break;
    }
    case GQPStrategy::GMRESSimpleIterativeSolver:
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      _buildKKTSimpleIterativeSolver();
      break;
    }
    case GQPStrategy::GMRESConePrecond: {
      _buildKKTPreconditionedGMRES_LDLT_ConePrecond();
      break;
    }
    }
  }

  void _commonKKTConstructionWithProductUpdate(T muIn, VecRef<T> x,
                                               VecRef<T> zPrev) {
    isize n = this->dim;
    isize m_eq = this->n_eq;

    isize off = 0;
    for (auto const &ineq : this->inequalityConstraints) {
      isize dimC = ineq.dScaled.size();
      Vec<T> arg =
          muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
      auto JC = ineq.cone.fastMultByDualJacobian(arg, ineq.CScaled);

      kkt.block(n + m_eq + off, 0, dimC, n) = JC;
      kkt.block(0, n + m_eq + off, n, dimC) = JC.transpose();
      off += dimC;
    }
  }

  void _kktConstructionInUpdateBase(T muIn, VecRef<T> x, VecRef<T> zPrev) {
    _commonKKTConstructionWithProductUpdate(muIn, x, zPrev);
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

  void _kktConstructionConePrecond(T muIn, VecRef<T> x, VecRef<T> zPrev) {
    _commonKKTConstructionWithProductUpdate(muIn, x, zPrev);
  }

  void _kktConstructionBaseSparseLDLT(T muIn, VecRef<T> x, VecRef<T> zPrev) {
    _commonKKTConstructionWithProductUpdate(muIn, x, zPrev);
  }

  void _constructionUpdate(T muIn, VecRef<T> x, VecRef<T> zPrev,
                           GQPStrategy strategy = GQPStrategy::Base) {

    // If we don't do in too time we get a bug due to aliasing
    kkt.topLeftCorner(this->dim, this->dim) += -oldCurvature;
    kkt.topLeftCorner(this->dim, this->dim) += curvature();
    switch (strategy) {
    case GQPStrategy::BaseSparseLDLT: {
      _kktConstructionBaseSparseLDLT(muIn, x, zPrev);
      break;
    }
    case GQPStrategy::GMRESConePrecond: {
      _kktConstructionConePrecond(muIn, x, zPrev);
      break;
    }
    case GQPStrategy::BaseProxqpLike:
    case GQPStrategy::Base: {
      _kktConstructionInUpdateBase(muIn, x, zPrev);
      break;
    }
    case GQPStrategy::DL:
    case GQPStrategy::GMRESSimpleIterativeSolver:
    case GQPStrategy::BaseWithoutProduct: {
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      _kktConstructionInUpdateWithoutProduct(muIn, x, zPrev);
      if (this->strategyState.lastStep < 0 ||
          this->strategyState.lastStep > this->settings.dlStepLimit) {
        staleKKT = kkt;
      }

      break;
    }
    }
    }
  }

  void _solveUpdateKKTBaseWithoutProduct() { luSolver.compute(kkt); }

  void _factorizeSparseLDLT() {
    sparseKKT = convertToSparseMat(kkt);
    sparseLdLt.compute(sparseKKT);
  }

  void _solveUpdateKKTBaseWithSparseLDLT() { _factorizeSparseLDLT(); }

  void _solveUpdateKKTBaseProxqpLike(T muIn, VecRef<T> x, VecRef<T> zPrev) {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;
    isize kktDim = n + m_eq + m_in;

    isize nChanged = 0;
    isize off = 0;
    for (auto const &ineq : this->inequalityConstraints) {
      isize dimC = ineq.dScaled.size();
      Vec<T> arg =
          muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
      Vec<T> oldArg = muIn * zPrev.segment(off, dimC) +
                      ineq.CScaled * this->strategyState.lastXFactorise +
                      ineq.dScaled;
      for (isize k = 0; k < dimC; ++k) {
        if ((arg(k) > T(0)) != (oldArg(k) > T(0))) {
          ++nChanged;
        }
      }
      off += dimC;
    }

    if (nChanged > 0) {
      isize r = 2 * nChanged;
      Mat<T> W(kktDim, r);
      Vec<T> alpha(r);

      isize col = 0;
      off = 0;
      for (auto const &ineq : this->inequalityConstraints) {
        isize dimC = ineq.dScaled.size();
        Vec<T> arg =
            muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
        Vec<T> oldArg = muIn * zPrev.segment(off, dimC) +
                        ineq.CScaled * this->strategyState.lastXFactorise +
                        ineq.dScaled;

        Mat<T> JC = ineq.cone.fastMultByDualJacobian(arg, ineq.CScaled);

        for (isize k = 0; k < dimC; ++k) {
          bool newActive = (arg(k) > T(0));
          bool oldActive = (oldArg(k) > T(0));

          if (newActive != oldActive) {
            T sign = newActive ? T(1) : T(-1);
            Vec<T> deltaJC = sign * ineq.CScaled.row(k).transpose();

            W.col(col).head(n) = deltaJC;
            W.col(col).segment(n, m_eq).setZero();
            W.col(col).segment(n + m_eq, m_in).setZero();
            W.col(col)(n + m_eq + off + k) = T(1);
            alpha(col) = T(0.5);

            W.col(col + 1).head(n) = deltaJC;
            W.col(col + 1).segment(n, m_eq).setZero();
            W.col(col + 1).segment(n + m_eq, m_in).setZero();
            W.col(col + 1)(n + m_eq + off + k) = T(-1);
            alpha(col + 1) = T(-0.5);

            col += 2;
          }
        }
        off += dimC;
      }

      ldlRankUpdate(W, alpha);
    }

    this->strategyState.lastXFactorise = x;
  }

  void _solveUpdate(T muIn, VecRef<T> x, VecRef<T> zPrev,
                    GQPStrategy strategy = GQPStrategy::Base) {
    switch (strategy) {
    case GQPStrategy::BaseProxqpLike: {
      _solveUpdateKKTBaseProxqpLike(muIn, x, zPrev);
      break;
    }
    case GQPStrategy::BaseSparseLDLT: {
      _solveUpdateKKTBaseWithSparseLDLT();
      break;
    }
    case GQPStrategy::Base: {
      _solveUpdateKKTBase(muIn, x, zPrev);
      break;
    }
    case GQPStrategy::BaseWithoutProduct: {
      _solveUpdateKKTBaseWithoutProduct();
      break;
    }
    case GQPStrategy::DL: {
      if (this->strategyState.lastStep < 0) {
        _solveUpdateKKTBaseWithoutProduct();
      }
      break;
    }
    case GQPStrategy::GMRESConePrecond:
    case GQPStrategy::GMRESSimpleIterativeSolver:
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
    if (this->debug) {
      std::cout << "@KKT construction update time taken: "
                << kktConstructionInUpdateTimer.timeInMilliSeconds() << "ms"
                << std::endl;
    }

    solveUpdateTimer.start();
    _solveUpdate(muIn, x, zPrev, strategy);
    solveUpdateTimer.end();
    if (this->debug) {
      std::cout << "@KKT solver update time taken: "
                << solveUpdateTimer.timeInMilliSeconds() << "ms" << std::endl;
    }
  }

  void _updateBarrierParamsBase(T rho_old, T rho_new, T r_eq_old, T r_eq_new,
                                T r_in_old, T r_in_new) {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    _commonUpdateBarrierParamsWithProduct(rho_old, rho_new, r_eq_old, r_eq_new,
                                          r_in_old, r_in_new);

    auto stack = _makeStack();

    if (rho_old != rho_new) {
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
    staleKKT = kkt;
  }

  void _updateBarrierParamsSimpleIterativeSolver(T rho_old, T rho_new,
                                                 T r_eq_old, T r_eq_new,
                                                 T r_in_old, T r_in_new) {
    _commonUpdateBarrrierParamasWithoutProduct(rho_old, rho_new, r_eq_old,
                                               r_eq_new, r_in_old, r_in_new);
  }

  void _commonUpdateBarrierParamsWithProduct(T rho_old, T rho_new, T r_eq_old,
                                             T r_eq_new, T r_in_old,
                                             T r_in_new) {
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

  void _updateBarrierParamsBaseSparseLDLT(T rho_old, T rho_new, T r_eq_old,
                                          T r_eq_new, T r_in_old, T r_in_new) {
    _commonUpdateBarrierParamsWithProduct(rho_old, rho_new, r_eq_old, r_eq_new,
                                          r_in_old, r_in_new);
    _factorizeSparseLDLT();
  }

  void _updateBarrierParams(T rho_old, T rho_new, T r_eq_old, T r_eq_new,
                            T r_in_old, T r_in_new,
                            GQPStrategy strategy = GQPStrategy::Base) {

    switch (strategy) {
    case GQPStrategy::BaseSparseLDLT: {
      _updateBarrierParamsBaseSparseLDLT(rho_old, rho_new, r_eq_old, r_eq_new,
                                         r_in_old, r_in_new);
      break;
    }
    case GQPStrategy::BaseProxqpLike:
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
    case GQPStrategy::DL: {
      _updateBarrierParamsBaseWithoutProduct(rho_old, rho_new, r_eq_old,
                                             r_eq_new, r_in_old, r_in_new);
      break;
    }
    case GQPStrategy::GMRESSimpleIterativeSolver:
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      _updateBarrierParamsSimpleIterativeSolver(rho_old, rho_new, r_eq_old,
                                                r_eq_new, r_in_old, r_in_new);
      break;
    }
    case GQPStrategy::GMRESConePrecond: {
      _updateBarrierParamsPreconditionedGMRES_LDLT_ConePrecond(
          rho_old, rho_new, r_eq_old, r_eq_new, r_in_old, r_in_new);
      break;
    }
    }
  }

  void _solveKKTBase(VecRefMut<T> rhs) {
    auto stack = _makeStack();
    ldl.solve_in_place(rhs, stack);
  }

  void _solveKKTWithoutProductDL() {
    Vec<T> rhsCopy = rhsExpanded;
    Vec<T> fsol = luSolver.solve(rhsCopy);
    Vec<T> sSol = luSolver.solve((kkt - staleKKT) * fsol);
    rhsExpanded = fsol - sSol;
  }

  void _solveKKTWithoutProduct(VecRefMut<T> rhs, GQPStrategy strategy) {

    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    rhsExpanded.head(n + m_eq + m_in) = rhs;
    rhsExpanded.tail(2 * m_in).setZero();

    // std::cout << rhsExpanded << std::endl;

    switch (strategy) {
    case GQPStrategy::GMRESSimpleIterativeSolver:
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::SimpleIterativeSolver: {
      _solveKKTSimpleIterativeSolver(strategy);
      break;
    }
    case GQPStrategy::BaseWithoutProduct: {
      _solveKKTBaseWithoutProduct();
      break;
    }
    case GQPStrategy::DL: {
      _solveKKTWithoutProductDL();
      break;
    }
    case GQPStrategy::BaseProxqpLike:
    case GQPStrategy::BaseSparseLDLT:
    case GQPStrategy::GMRESConePrecond:
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

    isize maxIterations = kkt.rows() * 2;
    T tol = this->settings.iterativeEpsilon;

    switch (strategy) {
    case GQPStrategy::GMRESSimpleIterativeSolver: {
      gminresSolver.setMaxIterations(maxIterations);
      gminresSolver.setTolerance(tol);
      gminresSolver.compute(kkt);
      break;
    }
    default: {
      minresSolver.setMaxIterations(maxIterations);
      minresSolver.setTolerance(tol);
      minresSolver.compute(kkt);
    }
    }

    Vec<T> rhsCopy = rhsExpanded;
    switch (strategy) {
    case GQPStrategy::GMRESSimpleIterativeSolver: {
      rhsExpanded = gminresSolver.solve(rhsCopy);
      break;
    }
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
  void _solveKKTGMRESConePrecond(VecRefMut<T> rhs) {
    gmresLdltSolver.preconditioner() = ldltPrecondWrapper;
    gmresLdltSolver.compute(kkt);

    Vec<T> rhsCopy = rhs;
    rhs = gmresLdltSolver.solveWithGuess(rhsCopy, strategyState.prevSolution);

    if (this->debug) {
      std::cout << "iter: " << gmresLdltSolver.iterations() << std::endl;
    }

    strategyState.prevSolution = rhs;
  }
  void _setupPrecondLDLT(isize kktDim) {
    auto stackReq = precondLdlt.factorize_req(kktDim) |
                    precondLdlt.solve_in_place_req(kktDim);
    isize needed = stackReq.alloc_req();
    if (precondLdltStack.len() < needed) {
      precondLdlt.reserve_uninit(kktDim);
      precondLdltStack.resize_for_overwrite(needed);
    }
  }

  void _solveKKTBaseSparseLDLT(VecRefMut<T> rhs) {
    Vec<T> rhsCopy = rhs;
    rhs = sparseLdLt.solve(rhsCopy);
  }

  void _solveKKT(VecRefMut<T> rhs, GQPStrategy strategy = GQPStrategy::Base) {
    // We use the same api for all solveKKT i.e we only take the rhs that
    // correspond to -(-rStatStar,-rEq,-rCone) and it should be overwritten
    // with dx,dy,dz , even if the system ahs more variable in fact like in
    // the Without product the extenriio on ly care about dx,dy,dz
    switch (strategy) {
    case GQPStrategy::BaseSparseLDLT: {
      _solveKKTBaseSparseLDLT(rhs);
      break;
    }
    case GQPStrategy::BaseProxqpLike:
    case GQPStrategy::Base: {
      _solveKKTBase(rhs);
      break;
    }
    case GQPStrategy::GMRESConePrecond: {
      _solveKKTGMRESConePrecond(rhs);
      break;
    }
      // For strategy using the extended kkt without product
    case GQPStrategy::SimpleIterativeSolver:
    case GQPStrategy::GMRESSimpleIterativeSolver:
    case GQPStrategy::SimpleIterativeSolverWithWarmStart:
    case GQPStrategy::DL:
    case GQPStrategy::BaseWithoutProduct: {
      _solveKKTWithoutProduct(rhs, strategy);
      break;
    }
    }
  }

  void _buildKKTPreconditionedGMRES_LDLT_ConePrecond() {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;
    isize kktDim = n + m_eq + m_in;

    if (kkt.rows() != kktDim) {
      kkt.resize(kktDim, kktDim);
      rhs.resize(kktDim);
    }

    this->_setupPrecondLDLT(kktDim);

    kkt.setZero();
    kkt.topLeftCorner(n, n) = this->objectiveAggr.HScaled + curvature();
    kkt.topLeftCorner(n, n).diagonal().array() += rho;

    isize off = n;
    for (auto const &eq : this->equalityConstraints) {
      isize mi = eq.AScaled.rows();
      kkt.block(0, off, n, mi) = eq.AScaled.transpose();
      kkt.block(off, 0, mi, n) = eq.AScaled;
      off += mi;
    }

    kkt.diagonal().segment(n, m_eq).setConstant(-muEq);
    kkt.diagonal().segment(n + m_eq, m_in).setConstant(-muIn);

    if (m_in > 0) {
      isize offJ = 0;
      VecRef<T> xInit = this->solutionState.xScaled;
      VecRef<T> zPrevInit = this->solutionState.zScaled;
      for (auto const &ineq : this->inequalityConstraints) {
        isize dimC = ineq.dScaled.size();
        Vec<T> arg = muIn * zPrevInit.segment(offJ, dimC) +
                     ineq.CScaled * xInit + ineq.dScaled;
        Mat<T> JC = ineq.cone.fastMultByPrecondJacobian(arg, ineq.CScaled);
        kkt.block(n + m_eq + offJ, 0, dimC, n) = JC;
        kkt.block(0, n + m_eq + offJ, n, dimC) = JC.transpose();
        offJ += dimC;
      }
    }

    {
      auto stack = proxsuite::linalg::veg::dynstack::DynStackMut{
          proxsuite::linalg::veg::from_slice_mut, precondLdltStack.as_mut()};
      precondLdlt.factorize(kkt.transpose(), stack);
    }

    ldltPrecondWrapper.ldlt = &precondLdlt;
    ldltPrecondWrapper.stack_buf = &precondLdltStack;
    gmresLdltSolver.preconditioner() = ldltPrecondWrapper;
    gmresLdltSolver.setMaxIterations(2 * (n + m_eq + m_in));
    gmresLdltSolver.setTolerance(this->settings.iterativeEpsilon);
  }

  void _updateBarrierParamsCommonLogic(T rho_old, T rho_new, T r_eq_old,
                                       T r_eq_new, T r_in_old, T r_in_new) {
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    T delta = std::max(this->strategyState.lastStep, 0.0);

    if (rho_old != rho_new) {
      delta = std::max(delta, std::abs(rho_old - rho_new));
      kkt.topLeftCorner(n, n).diagonal().array() += rho_new - rho_old;
    }
    if (r_eq_old != r_eq_new) {
      delta = std::max(delta, std::abs(r_eq_old - r_eq_new));
      kkt.diagonal().segment(n, m_eq).setConstant(-r_eq_new);
    }
    if (r_in_old != r_in_new) {
      delta = std::max(delta, std::abs(r_in_new - r_in_old));
      kkt.diagonal().segment(n + m_eq, m_in).setConstant(-r_in_new);
    }
  }
  void _updateBarrierParamsPreconditionedGMRES_LDLT_ConePrecond(
      T rho_old, T rho_new, T r_eq_old, T r_eq_new, T r_in_old, T r_in_new) {

    _commonUpdateBarrierParamsWithProduct(rho_old, rho_new, r_eq_old, r_eq_new,
                                          r_in_old, r_in_new);
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    T delta = std::max(this->strategyState.lastStep, 0.0);

    delta = std::max(delta, std::abs(rho_old - rho_new));

    delta = std::max(delta, std::abs(r_eq_old - r_eq_new));

    delta = std::max(delta, std::abs(r_in_new - r_in_old));

    if (this->strategyState.currentOuter == 0 ||
        this->strategyState.lastStep >= this->settings.dlStepLimit) {

      if (this->debug) {
        std::cout << "!!!Barrier param update!!!" << std::endl;
      }

      {
        auto stack = proxsuite::linalg::veg::dynstack::DynStackMut{
            proxsuite::linalg::veg::from_slice_mut, precondLdltStack.as_mut()};
        precondLdlt.factorize(kkt.transpose(), stack);
      }
    }
    gmresLdltSolver.setMaxIterations(2 * (n + m_eq + m_in));
    gmresLdltSolver.setTolerance(this->settings.iterativeEpsilon);
  }
};

} // namespace dense

} // namespace proxgqp
} // namespace proxsuite

#endif
