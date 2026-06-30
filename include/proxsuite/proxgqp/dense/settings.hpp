
#ifndef PROXSUITE_PROXGQP_DENSE_SETTINGS_HPP
#define PROXSUITE_PROXGQP_DENSE_SETTINGS_HPP

#include <proxsuite/linalg/veg/internal/typedefs.hpp>
#include <proxsuite/proxqp/settings.hpp>

namespace proxsuite {
namespace proxgqp {
namespace dense {

using isize = proxsuite::linalg::veg::isize;

template <typename T> struct GQPSettings : proxsuite::proxqp::Settings<T> {

  T iterativeEpsilon;
  T penaltyReduction;
  T epsNewtonInit;
  T epsOuterInit;
  T lineSearchReduction;
  T armijoConstant;
  isize maxLineSearchIters;
  T min_search_step;
  T stepInCaseBelowMin;
  T rhoIncreaseFactor;
  T maxRho;
  T dlStepLimit;

  GQPSettings()
      : proxsuite::proxqp::Settings<T>(), penaltyReduction(0.1),
        epsNewtonInit(1e-3), epsOuterInit(1e-2), lineSearchReduction(0.5),
        armijoConstant(1e-4), maxLineSearchIters(20), min_search_step(1e-5),
        stepInCaseBelowMin(1), rhoIncreaseFactor(10), maxRho(1.0),
        iterativeEpsilon(1e-8), dlStepLimit(0.01) {}
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_PROXGQP_DENSE_SETTINGS_HPP */
