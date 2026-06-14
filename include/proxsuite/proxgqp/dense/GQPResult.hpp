#ifndef PROXSUITE_GQP_RESULT
#define PROXSUITE_GQP_RESULT

#include "fwd.hpp"

namespace proxsuite {
namespace proxgqp {
namespace dense {

template <typename T>
struct GQPResult {
  Vec<T> x, y, z;
  isize outerIters = 0;
  isize totalInnerIters = 0;
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite

#endif
