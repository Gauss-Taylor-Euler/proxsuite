#ifndef PROXSUITE_GQP_WRAPPER
#define PROXSUITE_GQP_WRAPPER

#include "GQPWithSolve.hpp"

namespace proxsuite {
namespace proxgqp {
namespace dense {

template <typename T>
struct GQP : GQPWithSolve<T> {
  GQP(isize dim) : GQPWithSolve<T>(dim) {}
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite

#endif
