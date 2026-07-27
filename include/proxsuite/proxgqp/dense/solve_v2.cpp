#ifndef PROXSUITE_GQP_WITH_SOLVE_V2
#define PROXSUITE_GQP_WITH_SOLVE_V2

#include "fwd.hpp"
#include <vector>

#include "GQPResult.hpp"
#include "proxsuite/proxgqp/dense/cones/LorentzCone.hpp"
#include "proxsuite/proxgqp/dense/cones/PositiveOrthantCone.hpp"
#include "proxsuite/proxgqp/dense/settings.hpp"

namespace proxsuite {
namespace proxgqp {
namespace dense {
enum class ConeKind { Lorentz, Orthant };

// A simple cone class for user to pass as parameter
// That is easily copiable, hence there will be no need
// In the binding of thing like keep alive
struct ParamCone {
  int dimC;
  ConeKind kind;
  ParamCone(int dimC, ConeKind kind) : dimC(dimC), kind(kind) {}
};

template <typename T> struct PrecondOutput {

  // Will contai information that will permit to reverse preconditioner
  // operation
  PrecondOutput<T>() {}
};

// New version more functional programming oriented of GQP
// With a rank update strategy that will use hyhound(later)
template <typename T> struct GQPSolveV2 {
public:
  // solve  min 0.5*<Hx,x>+ <g,x> s.t Cx+d<=_K 0 and Ax=b
  // The main objective of this function is being the user interface of the
  // solver and being the main orchestrator
  GQPResult<T> solveWithInit(Mat<T> H, Vec<T> g, Mat<T> A, Vec<T> b, Mat<T> C,
                             Vec<T> d, std::vector<ParamCone> cones, Vec<T> x,
                             Vec<T> y, Vec<T> z,
                             const GQPSettings<T> settings = GQPSettings<T>(),
                             const bool debug = true) {
    // Preconditioning logic
    const PrecondOutput<T> precondOutput =
        preconditione(H, g, A, b, C, d, x, y, z);

    // Getting the main loop parameters values
    T rho = settings.default_rho;
    T muEq = settings.default_mu_eq;
    T muIn = settings.default_mu_in;
    T epsNewton = settings.epsNewtonInit * muIn;
    T epsOuter =
        settings.epsOuterInit * std::pow(muIn, this->settings.alpha_bcl);

    const T epsAbs = settings.eps_abs;

    const isize max_iter = settings.max_iter;
    const isize max_iter_in = settings.max_iter_in;

    // for outer and inner count
    isize outerCnt = 0;
    isize innerCnt = 0;

    // So that we won't have to recalculate them
    const Mat<T> AA = A.transpose() * A;
    const std::vector<Mat<T>> CCS = getCCS(C, cones);
    const std::vector<T> CS = getCs(C, cones);
    const std::vector<T> dS = getDs(d, cones);

    // Variables for residue
    Vec<T> rCone;
    Vec<T> rEq;
    Vec<T> rStat;
    Vec<T> rStatStar;

    // Variables for infeasibility
    // constraint violation + stationary condition
    T ineqInfeas;
    T eqInfeas;
    T statInfeas;

    // Main outer loop
    for (isize outer = 0; outer < max_iter; ++outer) {
      outerCnt++;

      // Checking whetter we converged

      ineqInfeas = calculateIneqInfeas(CS, dS, x, cones);
      eqInfeas = calculateEqInfeas(A, b, x);
      statInfeas = calculateStatInfeas(H, g, A, C, x, y, z);
      if (ineqInfeas <= epsAbs && eqInfeas <= epsAbs && statInfeas <= epsAbs) {
        break;
      }

      // Setting the inner vectors with the outer values
      Vec<T> xInner = x;
      Vec<T> yInner = y;
      Vec<T> zInner = z;

      // Inner loop
      for (isize inner = 0; inner < max_iter_in; ++inner) {
        innerCnt++;
      }
    }

    // Unpreconditionement logic at the end of the loop
    unPreconditione(H, g, A, b, C, d, x, y, z, precondOutput);

    // Filling the result
    const GQPResult<T> result;

    result.x = x;
    result.y = y;
    result.z = z;
    if (outerCnt >= settings.max_iter) {
      result.status = GQPSolverStatus::GQP_MAX_ITER_REACHED;
    } else if (innerCnt >= settings.max_iter_in) {
      result.status = GQPSolverStatus::GQP_MAX_INNER_ITER_REACHED;
    } else {
      result.status = GQPSolverStatus::GQP_SOLVED;
    }

    result.outerIters = outerCnt;
    result.totalInnerIters = innerCnt;

    return result;
  }

private:
  // Input d in the cone product + cones
  // Return per cone d
  std::vector<Mat<T>> getCs(VecRef<T> d, std::vector<ParamCone> &cones) {

    isize off = 0;

    std::vector<Mat<T>> dS;

    for (auto &cone : cones) {
      isize dimC = cone.dimC;
      dS.push_back(d.segment(off, dimC));
      off += dimC;
    }

    return dS;
  }
  // Input C in the cone product + cones
  // Return per cone C
  std::vector<Mat<T>> getCs(MatRef<T> C, std::vector<ParamCone> &cones) {

    isize off = 0;

    std::vector<Mat<T>> CS;

    for (auto &cone : cones) {
      isize dimC = cone.dimC;
      CS.push_back(C.middleRows(off, dimC));
      off += dimC;
    }

    return CS;
  }

  // Input C in the cone product + cones
  // Return per cone C.TC
  std::vector<Mat<T>> getCCS(MatRef<T> C, std::vector<ParamCone> &cones) {

    isize off = 0;

    std::vector<Mat<T>> CCS;

    for (auto &cone : cones) {
      isize dimC = cone.dimC;
      CCS.push_back(C.middleRows(off, dimC).transpose() *
                    C.middleRows(off, dimC));
      off += dimC;
    }

    return CCS;
  }

  PrecondOutput<T> preconditione(MatRef<T> H, VecRef<T> g, MatRef<T> A,
                                 VecRef<T> b, MatRef<T> C, VecRef<T> d,
                                 VecRef<T> x, VecRef<T> y, VecRef<T> z) {

    // TODO (for now we assume simply identity preconditioner)
    return PrecondOutput<T>();
  }

  void unPreconditione(MatRef<T> H, VecRef<T> g, MatRef<T> A, VecRef<T> b,
                       MatRef<T> C, VecRef<T> d, VecRef<T> x, VecRef<T> y,
                       VecRef<T> z, PrecondOutput<T> precondOutput) {}

  T calculateStatInfeas(MatRef<T> H, VecRef<T> g, MatRef<T> A, MatRef<T> C,
                        VecRef<T> x, VecRef<T> y, VecRef<T> z) {
    return infNorm(H * x + g + A.transpose() * y + C.transpose() * z);
  }

  T calculateIneqInfeas(std::vector<Mat<T>> &CS, std::vector<Vec<T>> &dS,
                        VecRef<T> x, std::vector<ParamCone> &cones) {

    T ineqInfeas = 0;

    for (isize i = 0; i < cones.size(); i++) {
      Vec<T> Cxd = CS[i] * x + dS[i];
      ineqInfeas = max(ineqInfeas, infNorm(Cxd + project(-Cxd, cones[i])));
    }
    return ineqInfeas;
  }

  T calculateEqInfeas(MatRef<T> A, VecRef<T> b, VecRef<T> x) {
    return infNorm(A * x - b);
  }

  Cone<T> getConeImplementation(ParamCone cone) {
    switch (cone.kind) {
    case ConeKind::Lorentz: {
      return LorentzCone<T>(cone.dimC);
    }
    case ConeKind::Orthant: {
      return PositiveOrthantCone<T>(cone.dimC);
    }
    }
  }

  Vec<T> project(VecRef<T> x, ParamCone cone) {
    const auto _cone = getConeImplementation(cone);
    return _cone.dualProject(x);
  }

  T infNorm(VecRef<T> v) { return v.template lpNorm<Eigen::Infinity>(); }

  T sqNorm(VecRef<T> v) { return v.squaredNorm(); }
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite
#endif
