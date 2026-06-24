#ifndef PROXSUITE_GQP_STRATEGY
#define PROXSUITE_GQP_STRATEGY

namespace proxsuite {
namespace proxgqp {
namespace dense {

enum class GQPStrategy {
  Base,
  BaseWithoutProduct,
  SimpleIterativeSolver,
  SimpleIterativeSolverWithWarmStart
};

} // namespace dense
} // namespace proxgqp
} // namespace proxsuite

#endif
