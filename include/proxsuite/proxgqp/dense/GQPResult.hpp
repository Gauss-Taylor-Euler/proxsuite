#ifndef PROXSUITE_GQP_RESULT
#define PROXSUITE_GQP_RESULT

#include "fwd.hpp"

namespace proxsuite {
namespace proxgqp {
namespace dense {

enum struct GQPSolverStatus {
  GQP_SOLVED,
  GQP_MAX_ITER_REACHED,
  GQP_MAX_INNER_ITER_REACHED,
  GQP_NOT_RUN
};

template <typename T>
struct GQPResult {
  Vec<T> x, y, z;
  isize outerIters = 0;
  isize totalInnerIters = 0;

  GQPSolverStatus status = GQPSolverStatus::GQP_NOT_RUN;
  T pri_res = 0;
  T dua_res = 0;
  T mu_eq = 0;
  T mu_in = 0;
  T rho = 0;
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite

#endif
