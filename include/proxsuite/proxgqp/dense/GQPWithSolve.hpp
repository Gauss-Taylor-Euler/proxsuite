#ifndef PROXSUITE_GQP_WITH_SOLVE
#define PROXSUITE_GQP_WITH_SOLVE
#include "GQPLDLWrapper.hpp"
#include "GQPResult.hpp"
#include "fwd.hpp"
#include <algorithm>
#include <cmath>
#include <stdio.h>

namespace proxsuite {
namespace proxgqp {
namespace dense {

template <typename T> struct GQPWithSolve : GQPLDLWrapper<T> {

  static T _infNorm(VecRef<T> v) {
    return v.template lpNorm<Eigen::Infinity>();
  }

  static T _sqNorm(VecRef<T> v) { return v.squaredNorm(); }

  Vec<T> xPrevOuter, yPrevOuter, zPrevOuter;
  Vec<T> xIterate, yIterate, zIterate;
  Vec<T> rStat, rStatStar, rEq, rCone, rDMw;

  GQPWithSolve(isize dim)
      : GQPLDLWrapper<T>(dim), xPrevOuter(dim), yPrevOuter(0), zPrevOuter(0),
        xIterate(dim), yIterate(0), zIterate(0), rStatStar(dim), rStat(dim),
        rDMw(dim), rEq(0), rCone(0) {}

  void _resizeStateVectors() {
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    auto resizeIf = [](Vec<T> &v, isize s) {
      if (v.size() != s) {
        v.resize(s);
      }
    };

    resizeIf(yPrevOuter, m_eq);
    resizeIf(zPrevOuter, m_in);
    resizeIf(yIterate, m_eq);
    resizeIf(zIterate, m_in);
    resizeIf(rEq, m_eq);
    resizeIf(rCone, m_in);
    resizeIf(this->solutionState.y, m_eq);
    resizeIf(this->solutionState.yScaled, m_eq);
    resizeIf(this->solutionState.z, m_in);
    resizeIf(this->solutionState.zScaled, m_in);
  }

  void _computeKKTResiduals(T rho, T muEq, T muIn, VecRef<T> x, VecRef<T> xPrev,
                            VecRef<T> y, VecRef<T> yPrev, VecRef<T> z,
                            VecRef<T> zPrev) {
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    auto &H = this->objectiveAggr.HScaled;
    auto &g = this->objectiveAggr.gScaled;

    rStatStar.noalias() = H * x;
    rStatStar += g;
    rStatStar.array() += rho * (x - xPrev).array();

    rStat.noalias() = H * x;
    rStat += g;
    rStat.array() += rho * (x - xPrev).array();

    rDMw.noalias() = H * x;
    rDMw += g;
    rDMw.array() += rho * (x - xPrev).array();

    if (m_eq > 0) {
      isize off = 0;
      for (auto const &eq : this->equalityConstraints) {
        isize mi = eq.AScaled.rows();

        rStatStar.noalias() += eq.AScaled.transpose() * y.segment(off, mi);

        rStat.noalias() += eq.AScaled.transpose() * y.segment(off, mi);

        rDMw.noalias() += eq.AScaled.transpose() * y.segment(off, mi);

        rEq.segment(off, mi).noalias() =
            muEq * (y.segment(off, mi) - yPrev.segment(off, mi)) -
            (eq.AScaled * x - eq.bScaled);

        rDMw.noalias() -=
            2 / m_eq * eq.AScaled.transpose() * rEq.segment(off, mi);

        off += mi;
      }
    }

    if (m_in > 0) {
      isize off = 0;
      for (auto const &ineq : this->inequalityConstraints) {
        isize dimC = ineq.dScaled.size();
        Vec<T> arg =
            muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
        rCone.segment(off, dimC).noalias() =
            muIn * z.segment(off, dimC) - ineq.cone.dualProject(arg);

        auto J = ineq.cone.dualJacobian(arg);
        auto JC = J * ineq.CScaled;

        rStatStar.noalias() += JC.transpose() * z.segment(off, dimC);

        rStat.noalias() += ineq.CScaled.transpose() * z.segment(off, dimC);

        rDMw.noalias() += JC.transpose() * z.segment(off, dimC);

        rDMw.noalias() -= 2 * JC.transpose() / m_in * rCone.segment(off, dimC);

        off += dimC;
      }
    }
  }
  T _meritKKT(T rho, T muEq, T muIn, VecRef<T> x, VecRef<T> xPrev,
              VecRef<T> yPrev, VecRef<T> zPrev) {
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;
    auto &H = this->objectiveAggr.HScaled;
    auto &g = this->objectiveAggr.gScaled;

    T phi =
        T(0.5) * x.dot(H * x) + g.dot(x) + T(0.5) * rho * _sqNorm(x - xPrev);

    if (m_eq > 0) {
      isize off = 0;
      for (auto const &eq : this->equalityConstraints) {
        isize mi = eq.AScaled.rows();
        Vec<T> axb = eq.AScaled * x - eq.bScaled;
        phi += yPrev.segment(off, mi).dot(axb) + _sqNorm(axb) / (T(2) * muEq);
        off += mi;
      }
    }

    if (m_in > 0) {
      isize off = 0;
      for (auto const &ineq : this->inequalityConstraints) {
        isize dimC = ineq.dScaled.size();
        Vec<T> arg =
            muIn * zPrev.segment(off, dimC) + ineq.CScaled * x + ineq.dScaled;
        Vec<T> proj = ineq.cone.dualProject(arg);
        phi +=
            (_sqNorm(proj) - muIn * muIn * _sqNorm(zPrev.segment(off, dimC))) /
            (T(2) * muIn);
        off += dimC;
      }
    }

    T M = phi;
    if (m_eq > 0) {
      M += _sqNorm(rEq) / (T(2) * muEq);
    }
    if (m_in > 0) {
      M += _sqNorm(rCone) / (T(2) * muIn);
    }
    return M;
  }

  T _primalInfeasibilityNorm(VecRef<T> x) {
    T p = T(0);
    for (auto const &eq : this->equalityConstraints) {
      Vec<T> axb = eq.AScaled * x;
      axb -= eq.bScaled;
      p = std::max(p, _infNorm(axb));
    }
    for (auto const &ineq : this->inequalityConstraints) {
      Vec<T> v = ineq.CScaled * x + ineq.dScaled;
      Vec<T> proj = ineq.cone.dualProject(-v);
      v += proj;
      p = std::max(p, _infNorm(v));
    }
    return p;
  }

  T _lineSearchArmijo(VecRef<T> newtonStep, T rho, T muEq, T muIn, T M0,
                      T dM_dw, VecRef<T> x, VecRef<T> y, VecRef<T> z,
                      VecRef<T> xPrev, VecRef<T> yPrev, VecRef<T> zPrev) {
    T lineSearchReduction = this->settings.lineSearchReduction;
    T armijoConstant = this->settings.armijoConstant;
    isize maxLineSearchIters = this->settings.maxLineSearchIters;
    T step = T(1);
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    Vec<T> xTrial = x;
    Vec<T> yTrial = y;
    Vec<T> zTrial = z;

    for (isize i = 0; i < maxLineSearchIters; ++i) {
      xTrial = x + step * newtonStep.head(n);
      if (m_eq > 0) {
        yTrial = y + step * newtonStep.segment(n, m_eq);
      }
      if (m_in > 0) {
        zTrial = z + step * newtonStep.tail(m_in);
      }

      _computeKKTResiduals(rho, muEq, muIn, xTrial, xPrev, yTrial, yPrev,
                           zTrial, zPrev);
      T M_trial = _meritKKT(rho, muEq, muIn, xTrial, xPrev, yPrev, zPrev);
      if (M_trial <= M0 + armijoConstant * step * dM_dw) {
        return step;
      }
      step *= lineSearchReduction;
    }
    return step;
  }

  bool _bclUpdate(T &muEq, T &muIn, T &epsNewton, T &epsOuter, isize outerIter,
                  T kktResidual, T primalInfeas, Vec<T> &x, Vec<T> &y,
                  Vec<T> &z, Vec<T> &xPrev, Vec<T> &yPrev, Vec<T> &zPrev) {
    auto const &s = this->settings;
    T penaltyReduction = s.penaltyReduction;
    T epsNewtonInit = s.epsNewtonInit;
    T epsOuterInit = s.epsOuterInit;

    if (primalInfeas <= s.eps_abs && kktResidual <= s.eps_abs) {
      xPrev = x;
      yPrev = y;
      zPrev = z;
      return true;
    }

    if (primalInfeas <= epsOuter || outerIter >= s.safe_guard) {
      xPrev = x;
      yPrev = y;
      zPrev = z;
      epsNewton = std::max(epsNewton * muIn, s.eps_abs);
      epsOuter = std::max(epsOuter * std::pow(muIn, s.beta_bcl), s.eps_abs);
    } else {
      y = yPrev;
      z = zPrev;

      T muEqNew = std::max(muEq * penaltyReduction, s.mu_min_eq);
      T muInNew = std::max(muIn * penaltyReduction, s.mu_min_in);
      this->_updateBarrierParams(s.default_rho, s.default_rho, muEq, muEqNew,
                                 muIn, muInNew);
      muEq = muEqNew;
      muIn = muInNew;

      epsNewton = epsNewtonInit * muIn;
      epsOuter = epsOuterInit * std::pow(muIn, s.alpha_bcl);
    }
    return false;
  }

  GQPResult<T> solve() {
    this->_readaptPreconditionementIfNeeded();
    _resizeStateVectors();

    isize outer = 0;
    isize inner = 0;
    isize totalInnerIters = 0;
    isize n = this->dim;
    isize m_eq = this->n_eq;
    isize m_in = this->n_in;

    if (m_eq > 0) {
      this->initSolutionWithEqualitySolution();
    } else {
      this->initSolutionWithZero();
    }
    this->_scaleSolution();

    xPrevOuter = xIterate = this->solutionState.xScaled;
    yPrevOuter = yIterate = this->solutionState.yScaled;
    zPrevOuter = zIterate = this->solutionState.zScaled;

    T rho = this->settings.default_rho;
    T muEq = this->settings.default_mu_eq;
    T muIn = this->settings.default_mu_in;
    T nRes = 0;
    T nResOrig = 0;
    T primalInfeas = 0;
    T epsNewtonInit = this->settings.epsNewtonInit;
    T epsOuterInit = this->settings.epsOuterInit;
    T epsNewton = epsNewtonInit * muIn;
    T epsOuter = epsOuterInit * std::pow(muIn, this->settings.alpha_bcl);

    this->_buildKKT();

    for (outer = 0; outer < this->settings.max_iter; ++outer) {

      for (inner = 0; inner < this->settings.max_iter_in; ++inner) {
        if (m_in > 0) {
          this->_updateFKBlocks(muIn, xIterate, zPrevOuter);
        }

        _computeKKTResiduals(rho, muEq, muIn, xIterate, xPrevOuter, yIterate,
                             yPrevOuter, zIterate, zPrevOuter);
        T nRes = _infNorm(rStat);
        if (m_eq > 0) {
          nRes = std::max(nRes, _infNorm(rEq));
        }
        if (m_in > 0) {
          nRes = std::max(nRes, _infNorm(rCone));
        }

        if (nRes <= epsNewton) {
          break;
        }

        T M0 = _meritKKT(rho, muEq, muIn, xIterate, xPrevOuter, yPrevOuter,
                         zPrevOuter);

        this->rhs.head(n) = -rStatStar;
        if (m_eq > 0) {
          this->rhs.segment(n, m_eq) = rEq;
        }
        if (m_in > 0) {
          this->rhs.tail(m_in) = rCone;
        }

        this->_solveKKT(this->rhs);

        auto dx = this->rhs.head(n);
        auto dy = this->rhs.segment(n, m_eq);
        auto dz = this->rhs.tail(m_in);

        T dM_dw = rDMw.dot(dx);

        if (m_eq > 0) {
          isize off = 0;
          for (auto const &eq : this->equalityConstraints) {
            isize mi = eq.AScaled.rows();
            Vec<T> Adx = eq.AScaled * dx;
            dM_dw -= 2 * rEq.segment(off, mi).dot(Adx) / muEq;
            dM_dw += rEq.segment(off, mi).dot(dy);
            off += mi;
          }
        }

        if (m_in > 0) {
          isize off = 0;
          for (auto const &ineq : this->inequalityConstraints) {
            isize dimC = ineq.dScaled.size();
            Vec<T> arg = muIn * zPrevOuter.segment(off, dimC) +
                         ineq.CScaled * xIterate + ineq.dScaled;
            Mat<T> J = ineq.cone.dualJacobian(arg);
            Vec<T> F_Cdx = J * (ineq.CScaled * dx);
            dM_dw -= 2 * rCone.segment(off, dimC).dot(F_Cdx) / muIn;

            dM_dw += rCone.segment(off, dimC).dot(dz);

            off += dimC;
          }
        }

        // printf("%e\n", dM_dw);

        T step = _lineSearchArmijo(this->rhs, rho, muEq, muIn, M0, dM_dw,
                                   xIterate, yIterate, zIterate, xPrevOuter,
                                   yPrevOuter, zPrevOuter);

        xIterate += step * this->rhs.head(n);
        if (m_eq > 0) {
          yIterate += step * this->rhs.segment(n, m_eq);
        }
        if (m_in > 0) {
          zIterate += step * this->rhs.tail(m_in);
        }

        ++totalInnerIters;
      }

      _computeKKTResiduals(rho, muEq, muIn, xIterate, xPrevOuter, yIterate,
                           yPrevOuter, zIterate, zPrevOuter);
      primalInfeas = _primalInfeasibilityNorm(xIterate);
      nRes = _infNorm(rStat);
      if (m_eq > 0) {
        nRes = std::max(nRes, _infNorm(rEq));
      }
      if (m_in > 0) {
        nRes = std::max(nRes, _infNorm(rCone));
      }

      nResOrig = _infNorm(rStat - rho * (xIterate - xPrevOuter));

      if (_bclUpdate(muEq, muIn, epsNewton, epsOuter, outer, nResOrig,
                     primalInfeas, xIterate, yIterate, zIterate, xPrevOuter,
                     yPrevOuter, zPrevOuter)) {
        break;
      }
    }

    this->solutionState.xScaled = xPrevOuter;
    this->solutionState.yScaled = yPrevOuter;
    this->solutionState.zScaled = zPrevOuter;
    this->_unscaleSolution();

    GQPResult<T> result;
    result.x = this->solutionState.x;
    result.y = this->solutionState.y;
    result.z = this->solutionState.z;
    result.outerIters = outer;
    result.totalInnerIters = totalInnerIters;

    if (outer >= this->settings.max_iter) {
      result.status = GQPSolverStatus::GQP_MAX_ITER_REACHED;
    } else if (inner >= this->settings.max_iter_in) {
      result.status = GQPSolverStatus::GQP_MAX_INNER_ITER_REACHED;
    } else {
      result.status = GQPSolverStatus::GQP_SOLVED;
    }
    result.pri_res = nResOrig;
    result.dua_res = primalInfeas;
    result.mu_eq = muEq;
    result.mu_in = muIn;
    result.rho = rho;
    return result;
  }
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite

#endif
